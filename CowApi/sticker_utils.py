import numpy as np


def extract_stickers(results, class_names):
    stickers = []

    for r in results:
        if r.boxes is None:
            continue

        for j, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]

            if "sticker" not in cls_name.lower():
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if r.masks is not None:
                poly = r.masks.xy[j]

                xs = poly[:, 0]
                ys = poly[:, 1]

                diameter = max(xs.max() - xs.min(),
                               ys.max() - ys.min())

                stickers.append({
                    "bbox": (x1, y1, x2, y2),
                    "diameter": diameter
                })

    return stickers


def match_sticker_to_cow(cow_box, stickers):
    x1, y1, x2, y2 = cow_box

    for s in stickers:
        sx1, sy1, sx2, sy2 = s["bbox"]

        if sx1 >= x1 and sy1 >= y1 and sx2 <= x2 and sy2 <= y2:
            return s["diameter"]

    return None


def get_cm_per_pixel(sticker_px, real_cm=15.0):
    if not sticker_px:
        return None

    return real_cm / sticker_px