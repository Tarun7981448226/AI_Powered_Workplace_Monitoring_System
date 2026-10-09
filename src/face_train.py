"""Step 2 - train the face recognition models from dataset/faces.

Trains two models from the photos captured by face_taker.py:
  * OpenCV LBPH      -> models/trainer_face.yml + models/label_map.json
  * PyTorch CNN      -> models/face_cnn.pt      + models/label_map_cnn.json

face_recognizer.py / main.py use the CNN when it exists, otherwise LBPH.
A CNN needs at least two people to learn from, so when only one person is
enrolled, a few public background faces are added automatically (they are
reported as "Unknown" and ignored by LBPH).
"""
import json
import os
import random

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ImportError:  # PyTorch missing -> LBPH only
    torch = None


# -----------------------------------------
# Paths
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "faces")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAINER_FILE = os.path.join(MODELS_DIR, "trainer_face.yml")
LABEL_MAP_FILE = os.path.join(MODELS_DIR, "label_map.json")
CNN_FILE = os.path.join(MODELS_DIR, "face_cnn.pt")
CNN_LABEL_FILE = os.path.join(MODELS_DIR, "label_map_cnn.json")

BACKGROUND_PREFIX = "_background"
IMG_SIZE = 96  # CNN input size


def list_people(dataset_path):
    return [p for p in sorted(os.listdir(dataset_path))
            if os.path.isdir(os.path.join(dataset_path, p))]


# =========================================================
# LBPH (classical OpenCV model)
# =========================================================
def train_lbph(dataset_path):
    print("\n[INFO] Training LBPH model...")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8)

    faces, ids, label_map = [], [], {}

    for person in list_people(dataset_path):

        if person.startswith(BACKGROUND_PREFIX):
            continue  # background faces are only used by the CNN

        person_id = len(label_map)
        label_map[person_id] = person
        count = 0

        person_path = os.path.join(dataset_path, person)

        for name in os.listdir(person_path):
            if not name.endswith(".jpg"):
                continue

            try:
                img = Image.open(os.path.join(person_path, name)).convert("L")
            except OSError:
                print("[WARNING] Skipping corrupted image:", name)
                continue

            img = cv2.equalizeHist(np.array(img, "uint8"))
            faces.append(cv2.resize(img, (220, 220)))
            ids.append(person_id)
            count += 1

        print(f"   {person}: {count} images")

        if count < 15:
            print("[WARNING] Very few images for this person.")

    if not faces:
        print("[ERROR] No training images found.")
        return None

    recognizer.train(faces, np.array(ids))

    os.makedirs(MODELS_DIR, exist_ok=True)
    recognizer.write(TRAINER_FILE)

    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f)

    print(f"[INFO] LBPH model saved: {TRAINER_FILE}")
    return label_map


# =========================================================
# PyTorch CNN
# =========================================================
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


if torch is not None:

    class FaceCNN(nn.Module):
        """Small CNN for grayscale face identification (input 1 x 96 x 96)."""

        def __init__(self, num_classes, embedding_dim=128, dropout=0.4):
            super().__init__()
            self.features = nn.Sequential(
                conv_block(1, 32),     # 96 -> 48
                conv_block(32, 64),    # 48 -> 24
                conv_block(64, 128),   # 24 -> 12
                conv_block(128, 256),  # 12 -> 6
                nn.AdaptiveAvgPool2d(1),
            )
            self.embedding = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(256, embedding_dim),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, x):
            return self.classifier(self.embedding(self.features(x)))


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
            return (self.to_tensor(img) - 0.5) / 0.5, int(self.labels[i])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_face(gray):
    """Same normalisation at training and recognition time."""
    return cv2.equalizeHist(cv2.resize(gray, (IMG_SIZE, IMG_SIZE)))


def add_background_faces(count=6):
    """Adds public Olivetti faces as extra CNN classes (needs scikit-learn)."""
    from sklearn.datasets import fetch_olivetti_faces

    images = (fetch_olivetti_faces(shuffle=False).images * 255).astype(np.uint8)

    for p in range(count):
        folder = os.path.join(DATASET_DIR, f"{BACKGROUND_PREFIX}_{p + 1}")
        os.makedirs(folder, exist_ok=True)
        for k in range(10):
            face = cv2.equalizeHist(cv2.resize(images[p * 10 + k], (220, 220)))
            cv2.imwrite(os.path.join(folder, f"{k}.jpg"), face)

    print(f"[INFO] Added {count} public background identities for the CNN")


def load_cnn_dataset(dataset_path):
    images, labels, label_map = [], [], {}

    for person in list_people(dataset_path):
        idx = len(label_map)
        label_map[idx] = person
        person_path = os.path.join(dataset_path, person)

        for name in os.listdir(person_path):
            if not name.endswith(".jpg"):
                continue
            img = cv2.imread(os.path.join(person_path, name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(preprocess_face(img))
                labels.append(idx)

    return np.array(images, dtype=np.uint8), np.array(labels), label_map


def split_per_class(labels, val_fraction=0.25, seed=0):
    """Every person appears in both the training and validation sets."""
    rng = random.Random(seed)
    train_idx, val_idx = [], []

    for label in sorted(set(labels.tolist())):
        idx = [i for i, l in enumerate(labels) if l == label]
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_fraction)) if len(idx) > 1 else 0
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]

    return train_idx, val_idx


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def train_cnn(dataset_path, epochs=40, batch_size=32, lr=2e-3):
    if torch is None:
        print("[WARNING] PyTorch is not installed - skipping the CNN.")
        return None

    images, labels, label_map = load_cnn_dataset(dataset_path)

    if len(label_map) < 2:
        try:
            add_background_faces()
        except Exception as error:  # e.g. scikit-learn missing / offline
            print(f"[WARNING] Could not add background faces ({error}).")
            print("          The CNN needs 2+ people - using LBPH only.")
            return None
        images, labels, label_map = load_cnn_dataset(dataset_path)

    torch.manual_seed(0)
    device = get_device()
    print(f"\n[INFO] Training CNN on {len(images)} images, "
          f"{len(label_map)} identities, device={device}")

    train_idx, val_idx = split_per_class(labels)
    train_loader = DataLoader(FaceDataset(images[train_idx], labels[train_idx], augment=True),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FaceDataset(images[val_idx], labels[val_idx]),
                            batch_size=batch_size)

    model = FaceCNN(len(label_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)

        val_acc = accuracy(model, val_loader, device)
        if epoch == 1 or epoch % 5 == 0:
            print(f"   epoch {epoch:2d}/{epochs}  "
                  f"loss={total_loss / len(train_loader.dataset):.3f}  val_acc={val_acc:.0%}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "num_classes": len(label_map)}, CNN_FILE)
    with open(CNN_LABEL_FILE, "w") as f:
        json.dump(label_map, f)

    print(f"[INFO] CNN saved: {CNN_FILE}  (validation accuracy {val_acc:.0%})")
    return label_map


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    if not os.path.exists(DATASET_DIR):
        print("[ERROR] Dataset folder not found. Run face_taker.py first.")
        raise SystemExit(1)

    people = train_lbph(DATASET_DIR)

    if people:
        train_cnn(DATASET_DIR)

    print("\n[INFO] Training finished.")
