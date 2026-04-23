import cv2
import numpy as np
from ultralytics import YOLO
import os

# =====================
# LOAD MODELS (1 раз!)
# =====================
cow_model = YOLO("models/Yolo26l-seg.pt")
pose_model = YOLO("models/yolo26l-poseB2.pt")
cls_model  = YOLO("models/yolo11l-cls.pt")
tracker_file = "models/botsort_reid20.yaml"
class_names = cow_model.names


def pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def process_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))
    height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))

    # =====================
    # STORAGE
    # =====================
    cow_data = {}
    measurements = {}

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # =====================
        # TRACKING
        # =====================
        cow_results = cow_model.track(
            frame,
            persist=True,
            conf=0.5,
            verbose=False,
            tracker=tracker_file
        )

        # =====================
        # STICKERS
        # =====================
        sticker_results = cow_model.predict(frame, conf=0.5, verbose=False)

        stickers = []

        for sr in sticker_results:
            if sr.boxes is None:
                continue

            for j, box in enumerate(sr.boxes):
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]

                if "sticker" not in cls_name.lower():
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if sr.masks is not None:
                    poly = sr.masks.xy[j]
                    xs = poly[:, 0]
                    ys = poly[:, 1]

                    diameter = max(xs.max() - xs.min(),
                                   ys.max() - ys.min())

                    stickers.append({
                        "bbox": (x1, y1, x2, y2),
                        "diameter": diameter
                    })

        # =====================
        # PROCESS COWS
        # =====================
        for r in cow_results:

            if r.boxes is None:
                continue

            for i, box in enumerate(r.boxes):

                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]

                if "cow" not in cls_name.lower():
                    continue

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if track_id not in cow_data:
                    cow_data[track_id] = {
                        "sticker_px": None,
                        "keypoints": {},
                        "breed": None,
                        "breed_conf": 0.0
                    }

                if track_id not in measurements:
                    measurements[track_id] = {
                        "3-4": [],
                        "2-1": [],
                        "0-6": []
                    }

                # =====================
                # MATCH STICKER
                # =====================
                for s in stickers:
                    sx1, sy1, sx2, sy2 = s["bbox"]

                    if sx1 >= x1 and sy1 >= y1 and sx2 <= x2 and sy2 <= y2:
                        cow_data[track_id]["sticker_px"] = s["diameter"]

                # =====================
                # POSE
                # =====================
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pose_results = pose_model.predict(crop, verbose=False)

                for pr in pose_results:
                    if pr.keypoints is None:
                        continue

                    kpts = pr.keypoints.xy.cpu().numpy()

                    for person in kpts:
                        for kp_id, (x, y) in enumerate(person):

                            if x <= 0 or y <= 0:
                                continue

                            xg = int(x + x1)
                            yg = int(y + y1)

                            cow_data[track_id]["keypoints"][kp_id] = (xg, yg)

                            cv2.circle(frame, (xg, yg), 3, (0,255,0), -1)

                # =====================
                # CLASSIFICATION
                # =====================
                if frame_idx % 10 == 0:

                    if (x2 - x1) > 80 and (y2 - y1) > 80:

                        cls_results = cls_model.predict(crop, verbose=False)

                        for cr in cls_results:
                            probs = cr.probs
                            if probs is None:
                                continue

                            top1 = int(probs.top1)
                            conf = float(probs.top1conf)

                            if conf > cow_data[track_id]["breed_conf"]:
                                cow_data[track_id]["breed"] = cls_model.names[top1]
                                cow_data[track_id]["breed_conf"] = conf

                # =====================
                # DISTANCES
                # =====================
                sticker_px = cow_data[track_id]["sticker_px"]
                keypoints = cow_data[track_id]["keypoints"]

                if sticker_px is not None:

                    cm_per_pixel = 15.0 / sticker_px
                    pairs = [(3,4),(2,1),(0,6)]

                    for a, b in pairs:

                        if a in keypoints and b in keypoints:

                            dist_px = pixel_distance(keypoints[a], keypoints[b])

                            if a == 3:
                                dist_cm = dist_px * cm_per_pixel * 2
                                pair_name = "3-4"
                            elif a == 2:
                                dist_cm = dist_px * cm_per_pixel
                                pair_name = "2-1"
                            else:
                                dist_cm = dist_px * cm_per_pixel
                                pair_name = "0-6"

                            if 10 < dist_cm < 500:
                                measurements[track_id][pair_name].append(dist_cm)

                            mid_x = (keypoints[a][0] + keypoints[b][0]) // 2
                            mid_y = (keypoints[a][1] + keypoints[b][1]) // 2

                            cv2.putText(frame,
                                        f"{dist_cm:.1f} cm",
                                        (mid_x, mid_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0,255,255),
                                        2)

                # =====================
                # DRAW BOX
                # =====================
                breed = cow_data[track_id]["breed"]

                label = f"Cow {track_id}" if not breed else f"Cow {track_id} | {breed}"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # =====================
    # SAVE TXT
    # =====================
    txt_path = output_path.replace(".mp4", ".txt")

    with open(txt_path, "w") as f:
        for track_id, data in measurements.items():

            breed = cow_data.get(track_id, {}).get("breed", "unknown")
            f.write(f"Cow ID {track_id} ({breed})\n")

            for pair_name, values in data.items():
                if values:
                    avg = sum(values) / len(values)
                    f.write(f"{pair_name}: {avg:.2f} cm\n")

            f.write("\n")

    return txt_path