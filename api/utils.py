import numpy as np


def pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def median(values):
    if not values:
        return None
    return float(np.median(values))


# вставь сюда свою KS_TABLE_2D без изменений

def kluver_strauch_weight_2d(chest, length):
    # (вставь свою функцию без изменений)
    ...