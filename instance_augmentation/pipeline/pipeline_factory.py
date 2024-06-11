from typing import Union, List, Dict, Any

from instance_augmentation.pipeline.generation_pipeline import GenerationPipeline
from instance_augmentation.pipeline.preprocessors import BaseDataPreprocessor, ResizeDataPreprocessor
from instance_augmentation.pipeline.postprocessors import BasePostprocessor, AnnotationInpainter
from instance_augmentation.pipeline.image_generator import ImageGenerator
from instance_augmentation.pipeline.diffusion_pipeline import InpaintControlNetInference, InpaintSDXLAdapterInference
from instance_augmentation.pipeline.converters import BaseConverter, DUTSConverter, ImageSaver, YOLOConverter, MetadataSaver


def get_preprocessor(preprocessing: str, image_size: int):
    if preprocessing == "none":
        return BaseDataPreprocessor()
    elif preprocessing == "resize":
        return ResizeDataPreprocessor(image_size)
    raise ValueError(f"Unknown preprocessing method: {preprocessing}")


def get_postprocessor(postprocessing: str):
    if postprocessing == "none":
        return BasePostprocessor()
    elif postprocessing == "verify_annotations":
        return AnnotationInpainter()
    raise ValueError(f"Unknown postprocessing method: {postprocessing}")


def get_image_generator(
    generator: str,
    base_model: str,
    control_methods: Union[str, List[str]],
    generation_config: Dict[str, Any],
) -> ImageGenerator:
    if generator == "inpaint_controlnet":
        generator_inference = InpaintControlNetInference(base_model=base_model, control_methods=control_methods)
    elif generator == "inpaint_sdxl_adapter":
        generator_inference = InpaintSDXLAdapterInference(base_model=base_model, control_methods=control_methods)
    else:
        raise ValueError(f"Unknown generator: {generator}")
    return ImageGenerator(
        generator_inference=generator_inference, control=control_methods, generation_config=generation_config
    )


def get_generation_pipeline(
    preprocessing: str,
    image_size: int,
    postprocessing: str,
    generator: str,
    base_model: str,
    control_methods: Union[str, List[str]],
    generation_config: Dict[str, Any],
):
    preprocessor = get_preprocessor(preprocessing, image_size)
    postprocessor = get_postprocessor(postprocessing)
    image_generator = get_image_generator(generator, base_model, control_methods, generation_config)
    generation_pipeline = GenerationPipeline(
        preprocessor=preprocessor,
        image_generator=image_generator,
        postprocessor=postprocessor,
    )
    return generation_pipeline


def get_data_converter(saver: str, save_folder: str) -> BaseConverter:
    if saver == "save_metadata":
        return MetadataSaver(save_folder)
    elif saver == "save_images":
        return ImageSaver(save_folder)
    elif saver == "duts":
        return DUTSConverter(save_folder)
    elif saver == "yolo":
        return YOLOConverter(save_folder)
    raise ValueError(f"Unknown saver: {saver}")
