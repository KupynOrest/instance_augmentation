import random
from typing import List, Dict, Any, Union, Optional

import torch
from torch import autocast
from PIL import Image
import numpy as np
from compel import Compel, ReturnedEmbeddingsType
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
    T2IAdapter,
    MultiAdapter,
)
from controlnet_aux import HEDdetector, NormalBaeDetector, LineartDetector, PidiNetDetector
from diffusers.models.attention_processor import AttnProcessor2_0

from instance_augmentation.pipeline.entities import ControlMethod
from instance_augmentation.pipeline.pipeline_sdxl_inpaint_t2i import StableDiffusionXLInpaintAdapterPipeline


class DepthAnythingProcessor:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").cuda()

    def __call__(self, image: np.ndarray) -> Image.Image:
        return self._get_depth_image(image)

    def _get_depth_image(self, image: np.ndarray) -> Image.Image:
        image = Image.fromarray(image)
        inputs = self.image_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        return depth


DEPTH_PROCESSOR = DepthAnythingProcessor()

CONTROL_MODELS = {
    "hed": ControlMethod(
        model_path="lllyasviel/control_v11p_sd15_softedge",
        preprocessor=HEDdetector.from_pretrained("lllyasviel/Annotators"),
    ),
    "depth": ControlMethod(model_path="lllyasviel/control_v11f1p_sd15_depth", preprocessor=DEPTH_PROCESSOR),
    "t2i_depth": ControlMethod(
        model_path="TencentARC/t2i-adapter-depth-midas-sdxl-1.0", preprocessor=DEPTH_PROCESSOR,
    ),
    "t2i_lineart": ControlMethod(
        model_path="TencentARC/t2i-adapter-lineart-sdxl-1.0",
        preprocessor=LineartDetector.from_pretrained("lllyasviel/Annotators"),
    ),
    "t2i_sketch": ControlMethod(
        model_path="TencentARC/t2i-adapter-sketch-sdxl-1.0",
        preprocessor=PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
    ),
    "normal": ControlMethod(
        model_path="lllyasviel/control_v11p_sd15_normalbae",
        preprocessor=NormalBaeDetector.from_pretrained("lllyasviel/Annotators"),
    ),
}


def optimize_model(model, run_compile: bool = True):
    model.to(memory_format=torch.channels_last)
    model.set_attn_processor(AttnProcessor2_0())
    if run_compile:
        torch.compile(model, mode="reduce-overhead", fullgraph=True)
    return model


def get_inpaint_controlnet(base_model: str, control_methods: List[ControlMethod], torch_dtype=torch.float16):
    controlnet = [ControlNetModel.from_pretrained(x.model_path, torch_dtype=torch_dtype) for x in control_methods]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch_dtype, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.enable_xformers_memory_efficient_attention()
    optimize_model(pipe.unet, run_compile=True)
    return pipe


class InpaintControlNetInference:
    def __init__(self, base_model: str, control_methods: List[str]):
        control_methods = [CONTROL_MODELS[x] for x in control_methods]
        self.pipe = get_inpaint_controlnet(base_model, control_methods)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe = self.pipe.to("cuda")
        self.compel = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        mask: Image.Image,
        control_images: List[Image.Image],
        generation_config: Dict[str, Any],
        small_object: bool = False,
    ):
        control_scale = generation_config["condition_scale"]
        if small_object:
            control_scale = [0.1 for _ in control_images]
        with autocast("cuda"), torch.inference_mode():
            conditioning = self.compel.build_conditioning_tensor(prompt)
            output = self.pipe(
                prompt_embeds=conditioning,
                negative_prompt=negative_prompt,
                image=image,
                control_image=control_images,
                mask_image=mask,
                num_images_per_prompt=1,
                num_inference_steps=generation_config["num_inference_steps"],
                guidance_scale=random.choice(generation_config["guidance_scale"]),
                controlnet_conditioning_scale=control_scale,
            )
            return output


class InpaintSDXLAdapterInference:
    def __init__(self, base_model: str, control_methods: List[str]):
        control_methods = [CONTROL_MODELS[x] for x in control_methods]
        adapters = self._get_adapters(control_methods)
        scheduler = UniPCMultistepScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.pipe = StableDiffusionXLInpaintAdapterPipeline.from_pretrained(
            base_model,
            adapter=adapters,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

    def _get_adapters(self, control_methods: List[ControlMethod]) -> Union[T2IAdapter, MultiAdapter]:
        if len(control_methods) == 1:
            return T2IAdapter.from_pretrained(control_methods[0].model_path, torch_dtype=torch.float16).to("cuda")
        return MultiAdapter(
            [T2IAdapter.from_pretrained(x.model_path, torch_dtype=torch.float16).to("cuda") for x in control_methods]
        )

    def _get_prompts_embeddings(self, prompt: str, negative_prompt: Optional[str]) -> dict:
        res = dict()
        res["prompt_embeds"], res["pooled_prompt_embeds"] = self.compel(prompt)
        if negative_prompt is not None:
            res["negative_prompt_embeds"], res["negative_pooled_prompt_embeds"] = self.compel(negative_prompt)
        return res

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        mask: Image.Image,
        control_images: List[Image.Image],
        generation_config: Dict[str, Any],
        small_object: bool = False,
    ):
        control_scale = generation_config["condition_scale"]
        with autocast("cuda"), torch.inference_mode():
            conditioning = self._get_prompts_embeddings(prompt, negative_prompt)
            output = self.pipe(
                **conditioning,
                image=image,
                mask_image=mask,
                adapter_image=control_images,
                num_inference_steps=generation_config["num_inference_steps"],
                adapter_conditioning_scale=control_scale,
                guidance_scale=random.choice(generation_config["guidance_scale"]),
            )
            return output
