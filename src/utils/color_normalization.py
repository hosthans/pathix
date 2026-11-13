import cv2
import numpy as np
from typing import Optional

def macenko_normalization(img: np.ndarray, target_means: Optional[np.ndarray] = None, target_stds: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize an image's LAB channel statistics to target means/stds (Macenko-style)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    for channel, arr in enumerate([l, a, b]):
        mean = arr.mean() if target_means is None else target_means[channel]
        std = arr.std() if target_stds is None else target_stds[channel]
        arr = (arr - arr.mean()) / (arr.std() + 1e-8) * std + mean
        arr = np.clip(arr, 0, 255)
        if channel == 0:
            l = arr
        elif channel == 1:
            a = arr
        else:
            b = arr
    lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
