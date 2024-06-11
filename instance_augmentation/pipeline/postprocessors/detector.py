from typing import List, Tuple, Optional

import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import logging

from instance_augmentation.pipeline.entities import Annotation, GeneratedResult
from instance_augmentation.pipeline.utils import blend

IOU_THRESHOLD = 0.7
logger = logging.Logger(__name__)


def coco80_to_coco91_class():
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


class AnnotationInpainter:
    def __init__(self, inpaint_only_original: bool = False, exclude_categories: Optional[List[str]] = None):
        self.verifier = DetectionVerifier(exclude_categories=exclude_categories)
        self.inpaint_only_original = inpaint_only_original

    def __call__(self, generated_result: GeneratedResult, annotations: List[Annotation]) -> GeneratedResult:
        final_image = np.array(generated_result.original_image.copy())
        generated_annotations, original_annotations = self.verifier.match_annotations(
            np.array(generated_result.image), annotations
        )
        if original_annotations is None:
            return generated_result
        if not self.inpaint_only_original:
            for annotation in generated_annotations:
                final_image = blend(final_image, generated_result.rendered_instances, annotation.mask)
        for annotation in original_annotations:
            final_image = blend(final_image, np.array(generated_result.original_image), annotation.mask)
        generated_result.image = Image.fromarray(final_image)
        generated_result.valid_annotations = generated_annotations
        generated_result.original_annotations = original_annotations
        return generated_result


class DetectionVerifier:
    def __init__(self, exclude_categories: Optional[List[str]] = None):
        self.model = YOLO("yolov8x.pt")
        self.exclude_categories = exclude_categories
        logger.info(f"Exclude categories: {self.exclude_categories}")

    @staticmethod
    def _iou(box_1: np.array, box_2: np.array) -> float:
        # 1.get the coordinate of inters
        ixmin = max(box_1[0], box_2[0])
        ixmax = min(box_1[2], box_2[2])
        iymin = max(box_1[1], box_2[1])
        iymax = min(box_1[3], box_2[3])

        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)

        # 2. calculate the area of inters
        inters = iw * ih

        # 3. calculate the area of union
        uni = (
            (box_1[2] - box_1[0] + 1.0) * (box_1[3] - box_1[1] + 1.0)
            + (box_2[2] - box_2[0] + 1.0) * (box_2[3] - box_2[1] + 1.0)
            - inters
        )

        # 4. calculate the overlaps between pred_box and gt_box
        iou = inters / uni

        return iou

    def _is_match(self, annotation: Annotation, bboxes: np.ndarray, class_labels: List[int]) -> bool:
        if annotation.keep_original:
            return False
        x, y, x1, y1 = annotation.bbox
        bbox = np.array([x, y, x1, y1])
        ious = [self._iou(bbox, x) for x in bboxes]
        for label, iou in zip(class_labels, ious):
            if iou > IOU_THRESHOLD and label == annotation.category_id:
                return True
        return False

    def _filter_original_annotations(
        self, annotations_to_keep: List[Annotation], correct_annotations: List[Annotation]
    ) -> List[Annotation]:
        result = []
        for annotation in annotations_to_keep:
            if self._should_inpaint_back(annotation, correct_annotations) and not self._in_exclude_categories(annotation):
                result.append(annotation)
        return result

    def _in_exclude_categories(self, annotation: Annotation) -> bool:
        if self.exclude_categories is None:
            return False
        label = annotation.label.replace("+++", "").split(',')[0]
        return label in self.exclude_categories

    def _should_inpaint_back(self, annotation: Annotation, correct_annotations: List[Annotation]) -> bool:
        for correct_anno in correct_annotations:
            iou_overlap = self._iou(annotation.bbox, correct_anno.bbox) > IOU_THRESHOLD * 0.7
            if iou_overlap or self._inscribed_bbox(annotation.bbox, correct_anno.bbox):
                return True
        return False

    def _inscribed_bbox(self, bbox1, bbox2) -> bool:
        # Check if bbox1 is inscribed in bbox2
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        return x1 >= x3 and y1 >= y3 and x2 <= x4 and y2 <= y4

    def match_annotations(
        self, image: np.ndarray, annotations: List[Annotation]
    ) -> Tuple[List[Annotation], Optional[List[Annotation]]]:
        yolo_detections = self.model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[0].boxes
        bboxes = yolo_detections.xyxy.cpu().numpy()
        class_labels = yolo_detections.cls.cpu().numpy().astype(np.int32)
        class_labels = [coco80_to_coco91_class()[label] for label in class_labels]
        correct_generations = []
        wrong_generations = []
        for annotation in annotations:
            keep_annotation = self._is_match(annotation, bboxes, class_labels)
            if keep_annotation:
                correct_generations.append(annotation)
            else:
                wrong_generations.append(annotation)
        if len(wrong_generations) == 0:
            return correct_generations, None

        wrong_generations = self._filter_original_annotations(wrong_generations, correct_generations)
        return correct_generations, wrong_generations


def is_similar(image1, image2):
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())
