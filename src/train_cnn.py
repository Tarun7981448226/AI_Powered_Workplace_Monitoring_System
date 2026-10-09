import cv2
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cnn_model import FaceCNN, IMG_SIZE, get_device


# -----------------------------------------
# Paths
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("FACE_MODELS_DIR", os.path.join(BASE_DIR, "models"))

CNN_FILE = os.path.join(MODELS_DIR, "face_cnn.pt")
CNN_LABEL_FILE = os.path.join(MODELS_DIR, "label_map_cnn.json")


def preprocess_face(gray):
    """Same normalisation used at training and inference time."""
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = cv2.equalizeHist(gray)
    return gray


# -----------------------------------------
# Data
# -----------------------------------------
def load_dataset(dataset_path):
    """Read dataset/faces/<person>/*.jpg -> (images, labels, label_map)."""
    images, labels, label_map = [], [], {}

    for idx, person in enumerate(sorted(os.listdir(dataset_path))):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        label_map[idx] = person

        for name in os.listdir(person_path):
            if not name.endswith(".jpg"):
                continue
            img = cv2.imread(os.path.join(person_path, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("[WARNING] Skipping corrupted image:", name)
                continue
            images.append(preprocess_face(img))
            labels.append(idx)

    return np.array(images, dtype=np.uint8), np.array(labels), label_map


def split_per_class(labels, val_fraction=0.25, seed=0):
    """Stratified split so every person appears in both train and validation."""
    rng = random.Random(seed)
    train_idx, val_idx = [], []

    for label in sorted(set(labels.tolist())):
        idx = [i for i, l in enumerate(labels) if l == label]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]

    return train_idx, val_idx


class FaceDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.to_tensor = transforms.ToTensor()
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomHorizontalFlip(),
        ]) if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment is not None:
            img = np.array(self.augment(img))
        x = self.to_tensor(img)
        x = (x - 0.5) / 0.5
        return x, int(self.labels[i])


# -----------------------------------------
# Train / evaluate
# -----------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train_model(train_imgs, train_labels, val_imgs, val_labels, num_classes,
                epochs=40, batch_size=32, lr=2e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    device = get_device()

    train_loader = DataLoader(FaceDataset(train_imgs, train_labels, augment=True),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FaceDataset(val_imgs, val_labels), batch_size=batch_size)

    model = FaceCNN(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)

        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        history.append((train_loss, val_acc))

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"[INFO] epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    return model.cpu(), history


def save_model(model, label_map, model_file=CNN_FILE, label_file=CNN_LABEL_FILE):
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(label_map)}, model_file)
    with open(label_file, "w") as f:
        json.dump(label_map, f)


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":

    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "dataset", "faces")

    if not os.path.exists(dataset_path):
        print("[ERROR] Dataset folder not found.")
        raise SystemExit(1)

    print("\n[INFO] Loading dataset...")
    images, labels, label_map = load_dataset(dataset_path)

    if len(images) == 0:
        print("[ERROR] No training images found.")
        raise SystemExit(1)

    if len(label_map) < 2:
        print("[ERROR] The CNN needs at least 2 classes (a classifier with one class "
              "would 'recognise' everyone).")
        print("        Enroll another person, or run: python src/add_background_faces.py")
        print("        (the app keeps working with the LBPH model in the meantime).")
        raise SystemExit(1)

    print(f"[INFO] {len(images)} images, {len(label_map)} people, device={get_device()}")

    train_idx, val_idx = split_per_class(labels)
    model, history = train_model(images[train_idx], labels[train_idx],
                                 images[val_idx], labels[val_idx], len(label_map))

    save_model(model, label_map)

    print(f"\n[INFO] Final validation accuracy: {history[-1][1]:.3f}")
    print(f"[INFO] Model saved: {CNN_FILE}")
