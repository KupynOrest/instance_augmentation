import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor


class SDSafetyFilter:
    def __init__(self):
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, image, safety_type: str = "black"):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.safety_checker.device)
        np_images, has_nsfw_concept = self.safety_checker(
            images=[image], clip_input=safety_checker_input.pixel_values.to(torch.float16)
        )
        return image, has_nsfw_concept[0]
