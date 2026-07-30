import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ImagePreprocessor:

    def __init__(self):

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess(self, image):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))

        image = image.convert("L")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        return tensor