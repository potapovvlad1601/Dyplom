import numpy as np


# =====================
# BASIC UTILS
# =====================
def pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def median(values):
    return float(np.median(values)) if len(values) > 0 else None


def get_cm_per_pixel(sticker_px):
    return 15.0 / sticker_px


# =====================
# GEOMETRY
# =====================
def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# =====================
# FEATURE ENGINEERING (MAIN)
# =====================
def extract_features(kpts, cow_seg, sticker_px):
    """
    kpts: (9,2)
    cow_seg: Nx2
    sticker_px: float
    """

    try:
        wither = kpts[0]
        pinbone = kpts[1]
        shoulder = kpts[2]

        front_top = kpts[3]
        front_bottom = kpts[4]

        rear_top = kpts[5]
        rear_bottom = kpts[6]

        h_top = kpts[7]
        h_bottom = kpts[8]

        # =====================
        # DISTANCES (px)
        # =====================
        body_len = pixel_distance(wither, pinbone)
        body_diag = pixel_distance(shoulder, pinbone)

        front_girth = pixel_distance(front_top, front_bottom)
        rear_girth = pixel_distance(rear_top, rear_bottom)
        height = pixel_distance(h_top, h_bottom)

        # =====================
        # SCALE
        # =====================
        if sticker_px is None or sticker_px < 1:
            return None

        cm_per_pixel = get_cm_per_pixel(sticker_px)

        body_len_cm = body_len * cm_per_pixel
        body_diag_cm = body_diag * cm_per_pixel * 2

        front_girth_cm = front_girth * cm_per_pixel
        rear_girth_cm = rear_girth * cm_per_pixel
        height_cm = height * cm_per_pixel

        # =====================
        # AREA
        # =====================
        cow_area_px = polygon_area(cow_seg)
        cow_area_cm2 = cow_area_px * (cm_per_pixel ** 2)

        # =====================
        # VOLUMES
        # =====================
        volume1 = body_len_cm * height_cm * front_girth_cm
        volume2 = body_len_cm * front_girth_cm * rear_girth_cm

        volume1_log = np.log(volume1 + 1e-6)
        volume2_log = np.log(volume2 + 1e-6)

        # =====================
        # RATIOS
        # =====================
        length_height_ratio = body_len_cm / (height_cm + 1e-6)
        girth_ratio = front_girth_cm / (rear_girth_cm + 1e-6)

        # =====================
        # BBOX FEATURES
        # =====================
        x_min, y_min = np.min(cow_seg, axis=0)
        x_max, y_max = np.max(cow_seg, axis=0)

        bbox_w = (x_max - x_min) * cm_per_pixel
        bbox_h = (y_max - y_min) * cm_per_pixel

        aspect_ratio = bbox_w / (bbox_h + 1e-6)

        bbox_area = bbox_w * bbox_h
        compactness = cow_area_cm2 / (bbox_area + 1e-6)

        # =====================
        # ANGLE
        # =====================
        angle = np.arctan2(
            pinbone[1] - wither[1],
            pinbone[0] - wither[0]
        )

        area_log = np.log(cow_area_cm2 + 1e-6)

        # =====================
        # FINAL VECTOR (ORDER IMPORTANT!)
        # =====================
        return np.array([
            body_len_cm,
            height_cm,
            front_girth_cm,
            rear_girth_cm,
            body_diag_cm,
            area_log,
            length_height_ratio,
            girth_ratio,
            volume1_log,
            volume2_log,
            bbox_w,
            bbox_h,
            aspect_ratio,
            compactness,
            angle
        ], dtype=np.float32)

    except:
        return None