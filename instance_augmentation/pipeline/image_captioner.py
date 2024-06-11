import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch


class ImageCaptioner:
    def __init__(self, question: str = "what is in the picture?", device: str = "cuda"):
        self.question = question
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-capfilt-large")
        self.model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-capfilt-large")

    def generate_caption(self, image: np.ndarray) -> str:
        inputs = self.processor(Image.fromarray(image), self.question, return_tensors="pt")
        # perform generation
        out = self.model.generate(**inputs)
        # postprocess result
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer
