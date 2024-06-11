from typing import Optional, List

import torch
import numpy as np
from PIL import Image
from groundingdino.util.inference import predict, load_model
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert

from pipeline.utils import get_relative_path

SAM_ENCODER_VERSION = "vit_h"
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]
BOX_TRESHOLD = 0.5
TEXT_TRESHOLD = 0.25


class SegmentationResult:
    def __init__(self, image: np.ndarray, bboxes: torch.Tensor, classes: List[str]):
        self.image = image
        self.classes = classes
        h, w, _ = image.shape
        bboxes = bboxes * torch.Tensor([w, h, w, h])
        self.bboxes = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        self.masks = None

    def __str__(self):
        return f"SegmentationResult(image={self.image.shape}, bboxes={self.bboxes}, classes={self.classes}, masks={self.masks})"


class InstanceSegmentationPredictor:
    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self.detection_model = load_model(get_relative_path("dino_config.py", __file__), get_relative_path("../models/groundingdino_swint_ogc.pth", __file__))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=get_relative_path("../models/sam_vit_h_4b8939.pth", __file__)).to(device=device)
        self.sam_predictor = SamPredictor(sam)
        self.class_names = class_names if class_names is not None else COCO_CLASSES

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image, None)
        return image_transformed

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=False
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def __call__(self, image: np.ndarray,  class_names: Optional[List[str]] = None, *args, **kwargs):
        if class_names is None:
            class_names = self.class_names
        class_names = " . ".join(class_names)
        boxes, _, classes = predict(
            model=self.detection_model,
            image=self.preprocess_image(image),
            caption=class_names,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        seg_result = SegmentationResult(image, boxes, classes)
        seg_result.masks = self.segment(
            image=image,
            xyxy=seg_result.bboxes
        )
        return seg_result
