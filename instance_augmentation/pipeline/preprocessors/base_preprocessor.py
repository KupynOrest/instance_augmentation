from typing import Tuple, List

from PIL import Image

from instance_augmentation.pipeline.entities import Annotation


class BaseDataPreprocessor:
    def __init__(self):
        pass

    def __call__(self, image: Image.Image, annotations: List[Annotation]) -> Tuple[Image.Image, List[Annotation]]:
        return image, annotations
