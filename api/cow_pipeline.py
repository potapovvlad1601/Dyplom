import cv2
import os
import joblib
import numpy as np
import torch

from .tracker import CowTracker
from .pose import PoseEstimator
from .classifier import CowClassifier
from .sticker_utils import *
from .utils import *


def process_video(video_path, output_dir, progress_callback=None):

    os.makedirs(output_dir, exist_ok=True)

    # =====================
    # MODELS
    # =====================
    if torch.cuda.is_available():
        device = "cuda"
        tracker = CowTracker("models/Yolo26l-seg.pt", "models/botsort_reid20.yaml", device)
        pose_model = PoseEstimator("models/yolo26l-poseB2.pt", device)
        classifier = CowClassifier("models/yolo11l-cls.pt", device)
    else:
        device = "cpu"
        tracker = CowTracker("models/Yolo26l-seg.pt", "models/botsort_reid20.yaml", device)
        pose_model = PoseEstimator("models/yolo26l-poseB2.pt", device)
        classifier = CowClassifier("models/yolo11l-cls.pt", device)

    # XGB MODEL
    weight_model = joblib.load("models/xgb_model_all2.2.pkl")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cow_data = {}
    measurements = {}
    weight_predictions = {}

    frame_idx = 0

    # =====================
    # MAIN LOOP
    # =====================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # progress
        if progress_callback and total_frames > 0:
            progress = int((frame_idx / total_frames) * 100)

            if frame_idx < total_frames * 0.3:
                stage = "tracking"
            elif frame_idx < total_frames * 0.7:
                stage = "pose"
            else:
                stage = "finalizing"

            progress_callback(progress, stage)

        # =====================
        # DETECTION + TRACKING
        # =====================
        results = tracker.track(frame)
        cows = tracker.get_cows(results)
        stickers = extract_stickers(results, tracker.class_names)

        for cow in cows:
            track_id = cow["id"]
            bbox = cow["bbox"]
            seg = cow.get("seg")   # ⚠️ важно чтобы tracker отдавал seg

            # init
            if track_id not in cow_data:
                cow_data[track_id] = {
                    "sticker_px": None,
                    "keypoints": {},
                    "breed": None,
                    "breed_conf": 0,
                    "seg": None
                }

            if track_id not in measurements:
                measurements[track_id] = {
                    "3-4": [],
                    "2-1": [],
                    "0-6": []
                }

            if track_id not in weight_predictions:
                weight_predictions[track_id] = []

            # =====================
            # SEGMENTATION
            # =====================
            if seg is not None:
                cow_data[track_id]["seg"] = seg

            # =====================
            # STICKER
            # =====================
            sticker_px = match_sticker_to_cow(bbox, stickers)
            if sticker_px:
                cow_data[track_id]["sticker_px"] = sticker_px

            # =====================
            # POSE
            # =====================
            keypoints = pose_model.get_keypoints(frame, bbox)
            cow_data[track_id]["keypoints"] = keypoints

            # =====================
            # BREED CLASSIFICATION
            # =====================
            if frame_idx % 10 == 0:
                breed, conf = classifier.classify(frame, bbox)
                if breed and conf > cow_data[track_id]["breed_conf"]:
                    cow_data[track_id]["breed"] = breed
                    cow_data[track_id]["breed_conf"] = conf

            # =====================
            # DISTANCES (legacy features)
            # =====================
            sticker_px = cow_data[track_id]["sticker_px"]

            if sticker_px and keypoints:
                cm_per_pixel = get_cm_per_pixel(sticker_px)

                pairs = [(3,4), (2,1), (0,6)]

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

            # =====================
            # XGB WEIGHT PREDICTION
            # =====================
            seg = cow_data[track_id].get("seg")

            if sticker_px and seg is not None and len(keypoints) >= 9:

                kpts_array = np.array(
                    [keypoints[i] for i in range(9)],
                    dtype=np.float32
                )

                features = extract_features(
                    kpts_array,
                    np.array(seg, dtype=np.float32),
                    sticker_px
                )

                if features is not None:
                    pred = weight_model.predict(features.reshape(1, -1))[0]
                    weight = float(np.exp(pred))  # log-space model

                    if 50 < weight < 1500:
                        weight_predictions[track_id].append(weight)

    cap.release()

    # =====================
    # FINAL AGGREGATION
    # =====================
    results_data = {}

    for track_id in cow_data.keys():

        breed = cow_data[track_id]["breed"]

        # measurements
        medians = {}
        for k, v in measurements.get(track_id, {}).items():
            m = median(v)
            if m:
                medians[k] = m

        # =====================
        # WEIGHT AGGREGATION
        # =====================
        wp = weight_predictions.get(track_id, [])

        if wp:
            wp = np.array(wp)
            med = np.median(wp)

            filtered = wp[np.abs(wp - med) < 0.2 * med]

            if len(filtered) > 0:
                final_weight = float(np.median(filtered))
            else:
                final_weight = float(med)
        else:
            final_weight = None

        results_data[track_id] = {
            "breed": breed,
            "measurements": medians,
            "weight": final_weight
        }

    return results_data