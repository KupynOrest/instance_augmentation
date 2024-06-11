import os
import copy
from typing import Optional, List, Union

from tqdm import tqdm

import cv2
from PIL import Image
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate

from instance_augmentation.pipeline.image_generator import ImageGenerator
from instance_augmentation.pipeline.generation_pipeline import GenerationPipeline
from instance_augmentation.pipeline.readers import BaseDatasetReader
from instance_augmentation.pipeline.converters import BaseConverter
from instance_augmentation.pipeline.pipeline_factory import get_generation_pipeline, get_data_converter


class DatasetGenerator:
    def __init__(
        self,
        dataset_reader: BaseDatasetReader,
        generation_pipeline: GenerationPipeline,
        converter: BaseConverter,
        config: Optional[DictConfig] = None,
        num_samples: int = 1
    ):
        self.dataset_reader = dataset_reader
        self.generation_pipeline = generation_pipeline
        self.converter = converter
        self.config = config
        self.num_samples = num_samples

    @staticmethod
    def read_rgb_image(image_path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    def run(self) -> None:
        for dataset_item in tqdm(self.dataset_reader):
            image = Image.fromarray(self.read_rgb_image(os.path.join(dataset_item.base_path, dataset_item.image_path)))
            for index in range(self.num_samples):
                annotations = [copy.deepcopy(annotation) for annotation in dataset_item.annotations]
                generation = self.generation_pipeline(image, annotations)
                if generation is not None:
                    self.converter.save_annotation(generation, dataset_item, index=index)

    @classmethod
    def from_config(cls, config: DictConfig) -> "DatasetGenerator":
        dataset_reader = instantiate(config["dataset_config"])
        image_generator_config = config["image_generator"]
        generator_inference = instantiate(
            image_generator_config["generator_inference"], control_methods=image_generator_config["control_methods"]
        )
        image_generator = ImageGenerator(
            generator_inference,
            control=image_generator_config["control_methods"],
            generation_config=image_generator_config["generation_config"],
        )
        generation_pipeline = GenerationPipeline(
            preprocessor=instantiate(config["preprocessor"]),
            image_generator=image_generator,
            postprocessor=instantiate(config["postprocessor"]),
        )
        converter = instantiate(config["saver"])
        num_samples = config.get("num_samples", 1)
        return cls(dataset_reader, generation_pipeline, converter, config, num_samples)

    @classmethod
    def from_params(
        cls,
        dataset_reader: BaseDatasetReader,
        save_folder: str,
        control_methods: Optional[Union[List[str], str]] = None,
        control_weights: Optional[Union[List[float], float]] = None,
        base_inpainting_model: str = "runwayml/stable-diffusion-inpainting",
        generator: str = "inpaint_controlnet",
        target_image_size: int = 768,
        num_inference_steps: int = 40,
        guidance_scales: Optional[List[float]] = None,
        preprocessing: str = "none",
        postprocessing: str = "none",
        data_saver: str = "save_metadata",
        num_samples: int = 1,
    ) -> "DatasetGenerator":
        """
        Args:
            dataset_reader: Dataset reader
            save_folder: Folder to save results
            control_methods: List of control methods to use, Available options: "depth", "hed", "normal", "t2i_sketch", "t2i_lineart", "t2i_depth"
            control_weights: List of control weights to use, the length should be equal to control_methods
            base_inpainting_model: Base model to use, default: "runwayml/stable-diffusion-inpainting"
            generator: Generator to use: Choose between "inpaint_controlnet", "inpaint_sdxl_adapter"
            target_image_size: Target image size
            num_inference_steps: Number of inference steps
            guidance_scales: Guidance scales to use, during the inference a random scale will be chosen from this list
            preprocessing: Preprocessing method to use, Available options: "none", "resize"
            postprocessing: Postprocessing method to use, Available options: "none", "verify_annotations".
            data_saver: Data saver to use, Available options: "save_metadata", "save_images", "yolo", "duts"
            num_samples: Number of augmented samples per one real image to generate
        """
        if control_methods is None:
            control_methods = ["depth", "hed"]
        if control_weights is None:
            control_weights = [0.7, 0.1]
        if guidance_scales is None:
            guidance_scales = [5.0, 7.5]
        generation_config = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scales,
            "condition_scale": control_weights,
            "image_size": target_image_size,
        }
        generation_pipeline = get_generation_pipeline(
            preprocessing=preprocessing,
            image_size=target_image_size,
            postprocessing=postprocessing,
            generator=generator,
            base_model=base_inpainting_model,
            control_methods=control_methods,
            generation_config=generation_config,
        )
        converter = get_data_converter(data_saver, save_folder)
        return cls(dataset_reader, generation_pipeline, converter, num_samples=num_samples)
