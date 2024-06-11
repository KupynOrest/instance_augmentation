from typing import List, Optional

from PIL import Image

from instance_augmentation.pipeline.entities import Annotation
from instance_augmentation.pipeline.image_generator import ImageGenerator
from instance_augmentation.pipeline.preprocessors import BaseDataPreprocessor
from instance_augmentation.pipeline.postprocessors import BasePostprocessor
from instance_augmentation.pipeline.entities import GeneratedResult


class GenerationPipeline:
    def __init__(
        self, preprocessor: BaseDataPreprocessor, image_generator: ImageGenerator, postprocessor: BasePostprocessor
    ):
        self.image_generator = image_generator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def __call__(self, image: Image.Image, annotations: List[Annotation]) -> Optional[GeneratedResult]:
        annotations_to_generate = [x for x in annotations if not x.keep_original]
        if len(annotations_to_generate) == 0:
            return None
        image, annotations = self.preprocessor(image, annotations)
        generation = self.postprocessor(
            self.image_generator(image, annotations_to_generate),
            annotations
        )
        return generation
