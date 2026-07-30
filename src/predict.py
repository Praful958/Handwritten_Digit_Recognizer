import torch
from src.model import DigitCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitCNN().to(device)

model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))

model.eval()