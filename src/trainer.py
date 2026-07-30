import torch
import torch.nn as nn
import torch.optim as optim
import json

from src.model import DigitCNN
from src.dataset import get_dataloaders
from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_PATH


class Trainer:

    def __init__(self):

        self.model = DigitCNN().to(DEVICE)

        self.train_loader, self.test_loader = get_dataloaders()

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        # Training History
        self.history = {
            "loss": [],
            "accuracy": []
        }


    def train(self):

        for epoch in range(EPOCHS):

            self.model.train()

            running_loss = 0
            correct = 0
            total = 0


            for images, labels in self.train_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)


                self.optimizer.zero_grad()


                outputs = self.model(images)


                loss = self.criterion(
                    outputs,
                    labels
                )


                loss.backward()

                self.optimizer.step()


                running_loss += loss.item()


                _, predicted = torch.max(
                    outputs,
                    1
                )


                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()



            epoch_loss = running_loss / len(self.train_loader)

            accuracy = 100 * correct / total


            # Save History
            self.history["loss"].append(
                epoch_loss
            )

            self.history["accuracy"].append(
                accuracy
            )


            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {accuracy:.2f}%"
            )



        # Save Model
        torch.save(
            self.model.state_dict(),
            MODEL_PATH
        )


        # Save Training History
        with open(
            "outputs/history.json",
            "w"
        ) as f:

            json.dump(
                self.history,
                f
            )


        print(
            f"✅ Model Saved: {MODEL_PATH}"
        )

        print(
            "✅ Training History Saved"
        )