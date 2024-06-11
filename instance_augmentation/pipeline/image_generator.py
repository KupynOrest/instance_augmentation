from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import PIL
from PIL import Image
import albumentations as A

from instance_augmentation.pipeline.entities import Annotation, GeneratedResult
from instance_augmentation.pipeline.diffusion_pipeline import CONTROL_MODELS
from instance_augmentation.pipeline.safety_checker import SDSafetyFilter
from instance_augmentation.pipeline.prompt_manager import PromptManager
from instance_augmentation.pipeline.utils import ensure_bbox_boundaries, extend_bbox

NEGATIVE_PROMPT = "3d render, cartoon, background, nsfw, nudity, 18+, naked, explicit content, uncensored"
AREA_THRESHOLD = 1000
MAX_RETRIES = 2


class ImageGenerator:
    def __init__(self, generator_inference, control: List[str], generation_config: Dict[str, Any]):
        self.control_methods = [CONTROL_MODELS[x] for x in control]
        self.inference_pipeline = generator_inference
        self.generation_config = generation_config
        self.resize_method = A.LongestMaxSize(generation_config["image_size"], interpolation=cv2.INTER_LANCZOS4)
        self.safety_filter = SDSafetyFilter()
        self.required_generate = generation_config.get("required_generate", False)
        self.prompt_manager = PromptManager(add_definition=generation_config.get("add_definition", True))

    def __call__(self, image: Image, annotations: List[Annotation]) -> GeneratedResult:
        if len(annotations) == 0:
            return GeneratedResult(original_image=image, image=image, nsfw=False)
        if annotations[0].mask is None:
            return self._process_from_bboxes(image, annotations)
        return self._process_from_masks(image, annotations)

    @staticmethod
    def _get_control_image(image: Image, control_method):
        control_image = control_method.preprocessor(np.array(image))
        if isinstance(control_image, np.ndarray):
            control_image = Image.fromarray(control_image)
        return control_image.resize(image.size, resample=PIL.Image.LANCZOS).convert("RGB")

    def _resize(self, image, mask):
        image = self.resize_method(image=np.array(image))["image"]
        mask = cv2.resize(np.array(mask), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if np.max(mask) > 1.0:
            mask = (mask > 128).astype(np.uint8)
        return Image.fromarray(image), mask

    @staticmethod
    def _blend(result: GeneratedResult, g_image: Image, mask: np.ndarray) -> None:
        if np.max(mask) > 1:
            mask = (mask > 128).astype(np.uint8)
        dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        blurred_mask = cv2.GaussianBlur(dilated_mask.copy().astype("float"), (5, 5), 0.0)[..., None]
        blended_image = (np.array(g_image) * blurred_mask) + (np.array(result.image) * (1 - blurred_mask))
        result.image = Image.fromarray(blended_image.astype(np.uint8))
        result.rendered_instances = (np.array(g_image) * dilated_mask[..., None]) + (
            result.rendered_instances * (1 - dilated_mask[..., None])
        )

    @staticmethod
    def _xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]

    def _process_from_masks(self, image: Image, annotations: List[Annotation]) -> GeneratedResult:
        result = GeneratedResult(original_image=image.copy(), image=image, nsfw=False)
        for index, annotation in enumerate(annotations):
            init_image, mask, crop_bbox = self._get_input_crops(result.image, annotation, len(annotations))
            g_image, nsfw_detected = self._generate_image(
                init_image, mask, self.prompt_manager(annotation.label), small_object=annotation.small_object
            )
            if nsfw_detected:
                result.nsfw = True
            if crop_bbox is not None:
                image.paste(g_image, (crop_bbox[0], crop_bbox[1]))
                g_image = image.copy()
            if len(annotations) > 1:
                self._blend(result, g_image, cv2.resize(annotation.mask, image.size, interpolation=cv2.INTER_NEAREST))
            else:
                result.rendered_instances = np.array(g_image)
                result.image = g_image
        result.image = result.image.resize(image.size, resample=PIL.Image.LANCZOS)
        return result

    def _get_input_crops(
        self, image: Image, annotation: Annotation, num_annotations: int
    ) -> Tuple[Image.Image, np.ndarray, Optional[List[float]]]:
        if annotation.bbox is None or num_annotations == 1:
            return image, annotation.mask, None
        x, y, w, h = ensure_bbox_boundaries(
            extend_bbox(self._xyxy_to_xywh(annotation.bbox), 0.9), np.array(image).shape
        )
        init_image = image.crop((x, y, x + w, y + h))
        mask = annotation.mask[y : y + h, x : x + w]
        crop_bbox = [x, y, w, h]
        return init_image, mask, crop_bbox

    def _generate_image(
        self, image: Image, mask: np.ndarray, prompt: str, small_object: bool = False
    ) -> Tuple[Image.Image, bool]:
        mask_image = self._build_mask(mask, dilate=not small_object)
        resized_image, mask_image = self._resize(image, mask_image)
        mask_pil_image = Image.fromarray(mask_image * 255)
        control_images = [self._get_control_image(resized_image, x) for x in self.control_methods]
        g_image, nsfw_detected = self._run_inference(
            resized_image, control_images, mask_pil_image, prompt, small_object=small_object
        )
        g_image = g_image.resize(image.size, resample=PIL.Image.LANCZOS)
        return g_image, nsfw_detected

    def _process_from_bboxes(self, image: Image, annotations: List[Annotation]) -> GeneratedResult:
        g_image = image.copy()
        bboxes = [x.bbox for x in annotations]
        control_images = [self._get_control_image(g_image, x) for x in self.control_methods]
        nsfw = False
        for index, annotation in enumerate(annotations):
            mask_image = self._build_mask_from_bbox(np.array(image), bboxes, index).resize(g_image.size)
            g_image, nsfw_detected = self._run_inference(
                g_image,
                control_images,
                mask_image,
                self.prompt_manager(annotation.label),
                small_object=annotation.small_object,
            )
            if nsfw_detected:
                nsfw = True
            g_image = g_image.resize(image.size, resample=PIL.Image.LANCZOS)
        return GeneratedResult(original_image=image, image=g_image, nsfw=nsfw)

    def _run_inference(
        self, image: Image, control: List[Any], mask: Image, prompt: str, small_object: bool = False
    ) -> Tuple[Image.Image, bool]:
        output = self.inference_pipeline(
            image, prompt, NEGATIVE_PROMPT, mask, control, self.generation_config, small_object=small_object
        )
        edited_image = output.images[0]
        _, nsfw_detected = self.safety_filter(np.array(edited_image))
        if nsfw_detected:
            if self.required_generate:
                return self._retry_generation(image, control, mask, prompt, small_object=small_object)
            return image, nsfw_detected
        return edited_image, False

    def _retry_generation(
        self, image: Image, control: List[Any], mask: Image, prompt: str, small_object: bool = False
    ) -> Tuple[Image.Image, bool]:
        result_image = image.copy()
        for _ in range(MAX_RETRIES):
            result_image = self.inference_pipeline(
                image, prompt, NEGATIVE_PROMPT, mask, control, self.generation_config, small_object=small_object
            ).images[0]
            _, nsfw_detected = self.safety_filter(np.array(result_image))
            if not nsfw_detected:
                return result_image, False
        return result_image, True

    @staticmethod
    def _build_mask(mask, dilate: bool = True):
        if not dilate:
            return mask
        dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
        return dilated_mask

    @staticmethod
    def _build_mask_from_bbox(image, bboxes, index):
        mask = np.zeros(image.shape[:2])
        x, y, x1, y1 = bboxes[index]
        mask[y:y1, x:x1] = 255
        for j_index, bbox in enumerate(bboxes):
            if j_index == index:
                continue
            x, y, x1, y1 = bboxes[j_index]
            mask[y:y1, x:x1] = 0
        return Image.fromarray(mask)
