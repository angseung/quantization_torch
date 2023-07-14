from typing import *
import numpy as np


def angle_between(
    p1: List[float], p2: List[float], signed: Optional[bool] = False
) -> float:
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

    return angle if signed else abs(angle)
