import cv2
import os

from .tracker import CowTracker
from .pose import PoseEstimator
from .classifier import CowClassifier
from .sticker_utils import *
from .utils import *


def process_video(video_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    tracker = CowTracker("models/yolo_seg.pt", "models/botsort.yaml")
    pose_model = PoseEstimator("models/yolo_pose.pt")
    classifier = CowClassifier("models/yolo_cls.pt")

    cap = cv2.VideoCapture(video_path)

    cow_data = {}
    measurements = {}

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = tracker.track(frame)
        cows = tracker.get_cows(results)
        stickers = extract_stickers(results, tracker.class_names)

        for cow in cows:
            track_id = cow["id"]
            bbox = cow["bbox"]

            if track_id not in cow_data:
                cow_data[track_id] = {
                    "sticker_px": None,
                    "keypoints": {},
                    "breed": None,
                    "breed_conf": 0
                }

            if track_id not in measurements:
                measurements[track_id] = {
                    "3-4": [],
                    "2-1": [],
                    "0-6": []
                }

            # sticker
            sticker_px = match_sticker_to_cow(bbox, stickers)
            if sticker_px:
                cow_data[track_id]["sticker_px"] = sticker_px

            # pose
            keypoints = pose_model.get_keypoints(frame, bbox)
            cow_data[track_id]["keypoints"] = keypoints

            # classification
            if frame_idx % 10 == 0:
                breed, conf = classifier.classify(frame, bbox)
                if breed and conf > cow_data[track_id]["breed_conf"]:
                    cow_data[track_id]["breed"] = breed
                    cow_data[track_id]["breed_conf"] = conf

            # distances
            sticker_px = cow_data[track_id]["sticker_px"]
            keypoints = cow_data[track_id]["keypoints"]

            if sticker_px and keypoints:
                cm_per_pixel = get_cm_per_pixel(sticker_px)

                pairs = [(3,4),(2,1),(0,6)]

                for a, b in pairs:
                    if a in keypoints and b in keypoints:

                        dist_px = pixel_distance(keypoints[a], keypoints[b])

                        if a == 3:
                            dist_cm = dist_px * cm_per_pixel * 2
                            name = "3-4"
                        elif a == 2:
                            dist_cm = dist_px * cm_per_pixel
                            name = "2-1"
                        else:
                            dist_cm = dist_px * cm_per_pixel
                            name = "0-6"

                        if 10 < dist_cm < 300:
                            measurements[track_id][name].append(dist_cm)

    cap.release()

    results_data = {}

    for track_id, data in measurements.items():
        breed = cow_data[track_id]["breed"]

        medians = {}
        for k, v in data.items():
            m = median(v)
            if m:
                medians[k] = m

        weight = kluver_strauch_weight_2d(
            medians.get("3-4"),
            medians.get("2-1")
        )

        results_data[track_id] = {
            "breed": breed,
            "measurements": medians,
            "weight": weight
        }

    return results_data