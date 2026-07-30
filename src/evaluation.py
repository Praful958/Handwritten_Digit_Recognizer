import torch
import numpy as np

from src.model import DigitCNN
from src.dataset import get_dataloaders
from config import DEVICE, MODEL_PATH


class Evaluator:

    def __init__(self):

        self.model = DigitCNN().to(DEVICE)

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )
        )

        self.model.eval()

        _, self.test_loader = get_dataloaders()


    def evaluate(self):

        correct = 0
        total = 0

        confusion = np.zeros(
            (10, 10),
            dtype=int
        )


        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)


                outputs = self.model(images)

                _, predicted = torch.max(
                    outputs,
                    1
                )


                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()


                for true, pred in zip(labels, predicted):

                    confusion[
                        true.item()
                    ][
                        pred.item()
                    ] += 1


        accuracy = 100 * correct / total


        return accuracy, confusion