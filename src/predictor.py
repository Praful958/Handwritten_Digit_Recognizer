import torch
import torch.nn.functional as F

from src.model import DigitCNN
from src.preprocess import ImagePreprocessor
from config import DEVICE, MODEL_PATH


class Predictor:

    def __init__(self):

        self.model = DigitCNN().to(DEVICE)

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )
        )

        self.model.eval()

        self.preprocessor = ImagePreprocessor()


    def predict(self, image):

        image_tensor = self.preprocessor.preprocess(
            image
        )

        image_tensor = image_tensor.to(
            DEVICE
        )


        with torch.no_grad():

            output = self.model(
                image_tensor
            )

            probabilities = F.softmax(
                output,
                dim=1
            )


            confidence, prediction = torch.topk(
                probabilities,
                3
            )


        results = []


        for i in range(3):

            results.append(
                {
                    "digit": prediction[0][i].item(),
                    "confidence": confidence[0][i].item()
                }
            )


        return results