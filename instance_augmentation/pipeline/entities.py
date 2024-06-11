from typing import Optional, List, Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from pytorch_toolbelt.utils import rle_to_string, rle_encode


@dataclass
class Annotation:
    label: str
    bbox: Optional[List[int]] = None  # xyxy
    mask: Optional[np.ndarray] = None
    category_id: Optional[int] = None
    mean_depth: Optional[float] = None
    small_object: bool = False
    keep_original: bool = False
    area: Optional[float] = None

    def to_dict(self):
        if self.mask is not None:
            mask_threshold = 128 if np.max(self.mask) > 1.0 else 0.5
            mask = (self.mask > mask_threshold).astype(np.uint8)
            mask = rle_to_string(rle_encode(mask))
        else:
            mask = []
        return {
            "label": self.label,
            "bbox": self.bbox if self.bbox is not None else [],
            "mask": mask,
            "category_id": self.category_id if self.category_id is not None else -1,
            "mean_depth": self.mean_depth if self.mean_depth is not None else -1,
            "small_object": int(self.small_object),
        }

    def __str__(self):
        return f"Annotation(label={self.label}, bbox={self.bbox}, category_id={self.category_id})"


@dataclass
class DatasetItem:
    base_path: str
    image_path: str
    annotations: List[Annotation]
    split: Optional[str] = None
    annotation_path: Optional[str] = None
    index: Optional[int] = None


@dataclass
class GeneratedResult:
    original_image: Image.Image
    image: Image.Image
    nsfw: bool = False
    rendered_instances: np.ndarray = field(init=False)
    valid_annotations: Optional[List[Annotation]] = None
    original_annotations: Optional[List[Annotation]] = None

    def __post_init__(self) -> None:
        self.rendered_instances = np.zeros(np.array(self.image).shape, dtype=np.uint8)


@dataclass
class ControlMethod:
    model_path: str
    preprocessor: Callable
