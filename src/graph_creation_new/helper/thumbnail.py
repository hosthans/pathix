import numpy as np
import cv2

def safe_thumbnail(slide, max_size: int) -> np.ndarray:
    """Generate an RGB thumbnail of a slide capped to a maximum dimension size."""
    w0, h0 = slide.dimensions

    scale = max(w0, h0) / float(max_size)
    if scale < 1.0:
        scale = 1.0

    tw, th = int(round(w0 / scale)), int(round(h0 / scale))
    tw = max(64, tw)
    th = max(64, th)

    best_level = int(np.argmin([
        abs(max(d) - max_size) for d in slide.level_dimensions
    ]))

    lw, lh = slide.level_dimensions[best_level]

    img = slide.read_region((0, 0), best_level, (lw, lh))
    arr = np.asarray(img)

    if arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

    if (lw, lh) != (tw, th):
        arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    return arr.astype(np.uint8)