from typing import List

from instance_augmentation.pipeline.entities import GeneratedResult, Annotation


class BasePostprocessor:
    def __init__(self):
        pass

    def __call__(self, generated_result: GeneratedResult, annotations: List[Annotation]) -> GeneratedResult:
        generated_result.valid_annotations = [x for x in annotations if not x.keep_original]
        return generated_result
