import random

from nltk.corpus import wordnet


PROMPTS = [
    "",
    "dlsr",
    "blurry edges",
    "natural",
    "ultrarealistic, 8k, uhd",
    "photorealistic",
    "sharp focus",
    "blurry background",
    "blurry foreground",
    "HDR, finely detailed",
]
LIGHTNING = [
    "accent lighting",
    "ambient lighting",
    "natural lighting",
    "dark lighting",
    "direct sunlight",
    "",
    "sunlight",
    "dramatic lighting",
    "soft lighting",
]
COLOR_CLASSES = [
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "boat",
    "bench",
    "bird",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "kite",
    "baseball bat",
    "skateboard",
    "surfboard",
    "chair",
    "couch",
    "bed",
    "dining table",
    "book",
    "vase",
]
COLORS = [
    "",
    ", white color",
    ", red color",
    ", green color",
    ", blue color",
    ", black color",
    ", yellow color",
    ", neutral color",
    ", light color",
    ", dark color",
]
BAD_DEFINITION_CLASSES = ["toaster", "microwave", "mouse", "toilet", "cake", "banana", "kite", "tv"]


class PromptManager:
    def __init__(self, add_definition: bool = True):
        self.add_definition = add_definition

    def __call__(self, label: str) -> str:
        syns = wordnet.synsets(label)
        if label in COLOR_CLASSES:
            label = f"({label})++{random.choice(COLORS)}"
        else:
            label = f"({label})++"
        if self.add_definition and len(syns) > 0 and label not in BAD_DEFINITION_CLASSES:
            word_definition = syns[0].definition()
            return f"{label}, {word_definition}, {random.choice(PROMPTS)}, {random.choice(LIGHTNING)}"
        return f"{label}, {random.choice(PROMPTS)}, {random.choice(LIGHTNING)}"
