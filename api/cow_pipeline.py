import cv2
import os
import numpy as np
import threading
import torch

from .LLM_api import build_weight_prompt_from_file, estimate_weight_from_files, save_weight_prompt
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

_MODELS = None
_MODELS_LOCK = threading.Lock()


def _build_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "tracker": CowTracker("models/Yolo26l-seg.pt", "models/botsort_reid20.yaml", device),
        "pose_model": PoseEstimator("models/yolo26l-poseB2.pt", device),
        "classifier": CowClassifier("models/yolo11l-cls.pt", device),
    }


def get_models():
    global _MODELS

    if _MODELS is None:
        with _MODELS_LOCK:
            if _MODELS is None:
                _MODELS = _build_models()

    return _MODELS


def preload_models():
    return get_models()


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

    # =====================
    # BBOX
    # =====================
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # =====================
    # ID + BREED (RED)
    # =====================
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

    # =====================
    # KEYPOINTS (RED DOTS)
    # =====================
    for point in keypoints.values():
        if point is None:
            continue
        px, py = map(int, point)
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

    # =====================
    # DISTANCES (YELLOW TEXT, NO LINES)
    # =====================
    for name, a, b, _, _ in DISTANCE_PAIRS:
        if a not in keypoints or b not in keypoints:
            continue

        pt1 = keypoints[a]
        pt2 = keypoints[b]

        if pt1 is None or pt2 is None:
            continue

        distance = distances.get(name)
        if distance is None:
            continue

        x1p, y1p = map(int, pt1)
        x2p, y2p = map(int, pt2)

        # midpoint
        mid_x = int((x1p + x2p) / 2)
        mid_y = int((y1p + y2p) / 2)

        cv2.putText(
            frame,
            f"{distance:.1f} cm",
            (mid_x + 5, mid_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),  # YELLOW
            2,
            cv2.LINE_AA,
        )


def _save_cow_snapshot(frame, bbox, track_id, frame_idx, output_dir, features):
    cows_dir = os.path.join(output_dir, "cows")
    os.makedirs(cows_dir, exist_ok=True)

    crop = crop_bbox(frame, bbox)
    if crop is None:
        return None

    base_name = f"cow_{track_id}_frame_{frame_idx}"
    image_rel_path = os.path.join("cows", f"{base_name}.jpg")
    txt_rel_path = os.path.join("cows", f"{base_name}.txt")

    image_path = os.path.join(output_dir, image_rel_path)
    txt_path = os.path.join(output_dir, txt_rel_path)

    if not cv2.imwrite(image_path, crop):
        return None

    save_features_txt(txt_path, features)

    return {
        "image_path": image_rel_path.replace("\\", "/"),
        "features_path": txt_rel_path.replace("\\", "/"),
        "frame_idx": frame_idx,
    }


def _resolve_output_path(output_dir, relative_path):
    if not relative_path:
        return None
    return os.path.join(output_dir, relative_path.replace("/", os.sep))


def _replace_extension(relative_path, new_extension):
    base_path, _ = os.path.splitext(relative_path)
    return f"{base_path}{new_extension}"


def process_video(video_path, output_dir, progress_callback=None):

    os.makedirs(output_dir, exist_ok=True)
    annotated_video_path = os.path.join(output_dir, "annotated.mp4")
    print("🔥 RUNNING UPDATED PIPELINE VERSION")
    # =====================
    # MODELS
    # =====================
    models = get_models()
    tracker = models["tracker"]
    pose_model = models["pose_model"]
    classifier = models["classifier"]

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
                    "latest_distances": {},
                    "snapshot_saved": False,
                    "snapshot_frame": None,
                    "snapshot_image": None,
                    "snapshot_features": None,
                }

            if track_id not in measurements:
                measurements[track_id] = {
                    "3-4": [],
                    "2-1": [],
                    "0-6": []
                }

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
            # ONE-TIME CENTER SNAPSHOT + FEATURE SAVE
            # =====================
            seg = cow_data[track_id].get("seg")
            if seg is None:
                seg = bbox_to_polygon(bbox)

            if (
                not cow_data[track_id]["snapshot_saved"]
                and sticker_px
                and all(i in keypoints for i in range(9))
                and is_bbox_centered(bbox, frame.shape)
            ):
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
                    snapshot = _save_cow_snapshot(
                        frame,
                        bbox,
                        track_id,
                        frame_idx,
                        output_dir,
                        features
                    )

                    if snapshot is not None:
                        cow_data[track_id]["snapshot_saved"] = True
                        cow_data[track_id]["snapshot_frame"] = snapshot["frame_idx"]
                        cow_data[track_id]["snapshot_image"] = snapshot["image_path"]
                        cow_data[track_id]["snapshot_features"] = snapshot["features_path"]

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

        results_data[track_id] = {
            "breed": breed,
            "measurements": medians,
            "snapshot_frame": cow_data[track_id]["snapshot_frame"],
            "snapshot_image": cow_data[track_id]["snapshot_image"],
            "snapshot_features": cow_data[track_id]["snapshot_features"],
            "snapshot_prompt": None,
            "weight": None,
            "weight_confidence": None,
            "weight_reasoning": None,
            "weight_model": None,
            "weight_error": None,
        }

        snapshot_image = _resolve_output_path(output_dir, cow_data[track_id]["snapshot_image"])
        snapshot_features = _resolve_output_path(output_dir, cow_data[track_id]["snapshot_features"])

        if snapshot_image and snapshot_features:
            try:
                prompt_relative_path = _replace_extension(
                    cow_data[track_id]["snapshot_features"],
                    "_prompt.txt"
                )
                prompt_output_path = _resolve_output_path(output_dir, prompt_relative_path)
                prompt_text = build_weight_prompt_from_file(snapshot_features)
                save_weight_prompt(prompt_output_path, prompt_text)
                results_data[track_id]["snapshot_prompt"] = prompt_relative_path

                llm_result = estimate_weight_from_files(
                    image_path=snapshot_image,
                    features_path=snapshot_features,
                )
                results_data[track_id]["weight"] = llm_result["weight"]
                results_data[track_id]["weight_confidence"] = llm_result["confidence"]
                results_data[track_id]["weight_model"] = llm_result["model"]
            except Exception as error:
                results_data[track_id]["weight_error"] = str(error)

    return results_data, annotated_video_path
