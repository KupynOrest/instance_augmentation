import os
from typing import Tuple, Union

import cv2
import numpy as np


KERNEL_SIZE = 7


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


def blend(image: np.ndarray, g_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.shape != g_image.shape:
        image = cv2.resize(image, (g_image.shape[1], g_image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    if mask.shape != g_image.shape[:2]:
        mask = cv2.resize(mask, (g_image.shape[1], g_image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    blurred_mask = cv2.GaussianBlur(mask.copy().astype("float"), (KERNEL_SIZE, KERNEL_SIZE), 0.0)[..., None]
    blended_image = (g_image * blurred_mask) + (image * (1 - blurred_mask))
    return blended_image.astype(np.uint8)


def extend_bbox(bbox: np.ndarray, offset: Union[Tuple[float, ...], float] = 0.1) -> np.ndarray:
    """
    Increases bbox dimensions by offset*100 percent on each side.

    IMPORTANT: Should be used with ensure_bbox_boundaries, as might return negative coordinates for x_new, y_new,
    as well as w_new, h_new that are greater than the image size the bbox is extracted from.

    :param bbox: [x, y, w, h]
    :param offset: (left, right, top, bottom), or (width_offset, height_offset), or just single offset that specifies
    fraction of spatial dimensions of bbox it is increased by.

    For example, if bbox is a square 100x100 pixels, and offset is 0.1, it means that the bbox will be increased by
    0.1*100 = 10 pixels on each side, yielding 120x120 bbox.

    :return: extended bbox, [x_new, y_new, w_new, h_new]
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset
    addition = 1.0 if (left + right + top + bottom) < 4.0 else 0.0
    return np.array([x - w * left, y - h * top, w * (addition + right + left), h * (addition + top + bottom)]).astype(
        int
    )


def ensure_bbox_boundaries(bbox: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Trims the bbox not the exceed the image.
    :param bbox: [x, y, w, h]
    :param img_shape: (h, w)
    :return: trimmed to the image shape bbox
    """
    x1, y1, w, h = bbox
    x1, y1 = min(max(0, x1), img_shape[1]), min(max(0, y1), img_shape[0])
    x2, y2 = min(max(0, x1 + w), img_shape[1]), min(max(0, y1 + h), img_shape[0])
    w, h = x2 - x1, y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")
