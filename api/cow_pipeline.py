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


DISTANCE_PAIRS = [
    ("3-4", 3, 4, 2.0, (0, 255, 255)),
    ("2-1", 2, 1, 1.0, (255, 255, 0)),
    ("0-6", 0, 6, 1.0, (255, 0, 255)),
]


def _draw_plain_text_lines(frame, lines, x, y, text_color=(0, 255, 255)):
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    line_gap = 8

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    total_h = sum(size[1] for size in sizes) + line_gap * (len(lines) - 1)
    max_w = max(size[0] for size in sizes)

    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(x, frame_w - max_w))
    y = max(total_h, min(y, frame_h))

    baseline_y = y - total_h
    for line, (_, text_h) in zip(lines, sizes):
        baseline_y += text_h
        cv2.putText(
            frame,
            line,
            (x, baseline_y),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_gap


def _draw_cow_overlay(frame, cow, state):
    x1, y1, x2, y2 = cow["bbox"]
    track_id = cow["id"]
    breed = state.get("breed") or "unknown"
    keypoints = state.get("keypoints") or {}
    distances = state.get("latest_distances") or {}

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(
        frame,
        f"ID {track_id} | {breed}",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),  # RED
        2,
        cv2.LINE_AA,
    )

    for point in keypoints.values():
        if point is None:
            continue
        px, py = map(int, point)
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

    distance_lines = []
    for name, _, _, _, _ in DISTANCE_PAIRS:
        distance = distances.get(name)
        if distance is None:
            continue
        distance_lines.append(f"{name}: {distance:.1f} cm")

    if distance_lines:
        preferred_y = y2 + 24 + len(distance_lines) * 22
        if preferred_y > frame.shape[0]:
            preferred_y = max(24 + len(distance_lines) * 22, y1 - 10)
        _draw_plain_text_lines(
            frame,
            distance_lines,
            x1,
            preferred_y,
            text_color=(0, 255, 255),
        )


def process_video(video_path, output_dir, progress_callback=None):

    os.makedirs(output_dir, exist_ok=True)
    annotated_video_path = os.path.join(output_dir, "annotated.mp4")

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
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    annotated_writer = cv2.VideoWriter(
        annotated_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not annotated_writer.isOpened():
        cap.release()
        raise ValueError(f"Unable to create output video: {annotated_video_path}")

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
        annotated_frame = frame.copy()

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
                    "seg": None,
                    "latest_distances": {}
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

                for name, a, b, multiplier, _ in DISTANCE_PAIRS:
                    if a in keypoints and b in keypoints:
                        dist_px = pixel_distance(keypoints[a], keypoints[b])
                        dist_cm = dist_px * cm_per_pixel * multiplier

                        if 10 < dist_cm < 300:
                            measurements[track_id][name].append(dist_cm)
                            cow_data[track_id]["latest_distances"][name] = dist_cm

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

            _draw_cow_overlay(annotated_frame, cow, cow_data[track_id])

        annotated_writer.write(annotated_frame)

    cap.release()
    annotated_writer.release()

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

    return results_data, annotated_video_path
