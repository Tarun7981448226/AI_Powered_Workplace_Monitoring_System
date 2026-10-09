import json
import os

import torch
import torch.nn.functional as F

from cnn_model import FaceCNN, IMG_SIZE, get_device
from train_cnn import CNN_FILE, CNN_LABEL_FILE, preprocess_face


class CNNFaceRecognizer:
    """Loads the trained FaceCNN and identifies a grayscale face crop."""

    def __init__(self, model_file=CNN_FILE, label_file=CNN_LABEL_FILE, threshold=0.7):
        with open(label_file, "r") as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

        checkpoint = torch.load(model_file, map_location="cpu")
        self.device = get_device()
        self.model = FaceCNN(checkpoint["num_classes"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.threshold = threshold

    @staticmethod
    def available():
        return os.path.exists(CNN_FILE) and os.path.exists(CNN_LABEL_FILE)

    @torch.no_grad()
    def predict(self, gray_face):
        """Returns (name, probability). name is 'Unknown' below the threshold."""
        face = preprocess_face(gray_face)
        x = torch.from_numpy(face).float().div(255).sub(0.5).div(0.5)
        x = x.view(1, 1, IMG_SIZE, IMG_SIZE).to(self.device)

        probs = F.softmax(self.model(x), dim=1)[0]
        prob, idx = probs.max(0)
        name = self.label_map.get(idx.item(), "Unknown")

        if prob.item() < self.threshold or name.startswith("_background"):
            name = "Unknown"
        return name, prob.item()
