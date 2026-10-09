"""Step 2 - train the face recognition models from dataset/faces.

Trains from the data captured by face_taker.py:
  * OpenCV LBPH      -> models/trainer_face.yml + models/label_map.json
  * PyTorch CNN      -> models/face_cnn.pt      + models/label_map_cnn.json   (faces)
  * PyTorch MLP      -> models/sign_net.pt                                    (hand signs)

Hand signs are learned from the public HaGRID gesture dataset (CC BY-SA 4.0,
https://github.com/hukenovs/hagrid): on first run it is downloaded (about 1 GB,
one time), MediaPipe extracts hand landmarks from ~12,000 images, and the
compact result is stored in dataset/signs/ - the images are not kept.

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

from face_recognizer import HAND_FILE, NO_SIGN_LABEL, SIGN_FEATURES, HandTracker

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
SIGN_DATA_FILE = os.path.join(BASE_DIR, "dataset", "signs", "hagrid_landmarks.npz")
SIGN_FILE = os.path.join(MODELS_DIR, "sign_net.pt")
SIGN_ZIP_URL = ("https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/"
                "resolve/main/hagrid-sample-30k-384p.zip")

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


    class SignNet(nn.Module):
        """MLP that classifies a hand pose from its 63 landmark coordinates."""

        def __init__(self, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(SIGN_FEATURES, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.net(x)


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
# Hand signs (public HaGRID dataset -> MediaPipe landmarks -> PyTorch MLP)
# =========================================================
def build_sign_dataset(zip_path=None, per_class=600, no_sign_count=1500, seed=0):
    """Extracts hand landmarks from the HaGRID sample and saves them as one small file."""
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    tmp = None
    if zip_path is None:
        tmp = tempfile.mkdtemp()
        zip_path = os.path.join(tmp, "hagrid.zip")
        print("[INFO] Downloading the HaGRID gesture dataset (about 1 GB, one time)...")

        def progress(blocks, size, total):
            if total > 0 and blocks % 2000 == 0:
                print(f"   {min(blocks * size / total, 1):.0%}")

        urllib.request.urlretrieve(SIGN_ZIP_URL, zip_path, reporthook=progress)

    root = "hagrid-sample-30k-384p"
    rng = random.Random(seed)
    tracker = HandTracker(static=True)
    features, labels, classes = [], [], []

    try:
        with zipfile.ZipFile(zip_path) as z:
            names = set(z.namelist())
            samples, no_sign = {}, []

            for name in sorted(names):
                if not (name.startswith(f"{root}/ann_train_val/") and name.endswith(".json")):
                    continue
                gesture = os.path.basename(name)[:-5]
                folder = f"{root}/hagrid_30k/train_val_{gesture}"
                annotations = json.loads(z.read(name))
                samples[gesture] = []

                for image_id, a in annotations.items():
                    path = f"{folder}/{image_id}.jpg"
                    if path not in names:
                        continue
                    for box, label in zip(a["bboxes"], a["labels"]):
                        if label == gesture:
                            samples[gesture].append((path, box))
                        elif label == NO_SIGN_LABEL:
                            no_sign.append((path, box))

            samples[NO_SIGN_LABEL] = no_sign
            classes = sorted(samples)
            print(f"[INFO] Extracting hand landmarks for {len(classes)} classes...")

            for idx, cls in enumerate(classes):
                items = samples[cls]
                rng.shuffle(items)
                limit = no_sign_count if cls == NO_SIGN_LABEL else per_class
                kept = 0

                for path, (bx, by, bw, bh) in items:
                    if kept >= limit:
                        break
                    img = cv2.imdecode(np.frombuffer(z.read(path), np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue

                    h, w = img.shape[:2]
                    m = 0.3 * max(bw, bh)  # context around the hand
                    x0, y0 = int(max(bx - m, 0) * w), int(max(by - m, 0) * h)
                    x1, y1 = int(min(bx + bw + m, 1) * w), int(min(by + bh + m, 1) * h)
                    if x1 - x0 < 20 or y1 - y0 < 20:
                        continue

                    feats, _ = tracker.process(img[y0:y1, x0:x1])
                    if feats is not None:
                        features.append(feats)
                        labels.append(idx)
                        kept += 1

                print(f"   {cls:16s} {kept} samples")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(os.path.dirname(SIGN_DATA_FILE), exist_ok=True)
    np.savez_compressed(SIGN_DATA_FILE, features=np.array(features, np.float32),
                        labels=np.array(labels), classes=np.array(classes))
    print(f"[INFO] Saved {len(features)} landmark samples: {SIGN_DATA_FILE}")


def augment_poses(x):
    """Random rotation, scale, mirror and jitter on a batch of poses (B, 63)."""
    batch = x.shape[0]
    pts = x.clone().view(batch, 21, 3)

    angle = (torch.rand(batch) * 2 - 1) * 0.26             # about +-15 degrees
    cos, sin = torch.cos(angle), torch.sin(angle)
    px, py = pts[:, :, 0].clone(), pts[:, :, 1].clone()
    pts[:, :, 0] = cos[:, None] * px - sin[:, None] * py
    pts[:, :, 1] = sin[:, None] * px + cos[:, None] * py

    flip = torch.rand(batch) < 0.5                         # left <-> right hand
    pts[flip, :, 0] *= -1

    pts *= 1 + 0.1 * (torch.rand(batch, 1, 1) * 2 - 1)
    pts += 0.01 * torch.randn_like(pts)
    return pts.view(batch, -1)


def train_signs(epochs=60, batch_size=128, lr=2e-3):
    if torch is None:
        print("[WARNING] PyTorch is not installed - skipping sign training.")
        return None

    if not os.path.exists(HAND_FILE):
        print("[WARNING] assets/hand_landmarker.task is missing - skipping signs.")
        return None

    if not os.path.exists(SIGN_DATA_FILE):
        try:
            build_sign_dataset()
        except Exception as error:  # e.g. offline
            print(f"[WARNING] Could not build the sign dataset ({error}) - skipping signs.")
            return None

    data = np.load(SIGN_DATA_FILE)
    x, y, classes = data["features"], data["labels"], [str(c) for c in data["classes"]]

    torch.manual_seed(0)
    train_idx, val_idx = split_per_class(y, val_fraction=0.2)
    xt, yt = torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])
    xv, yv = torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx])

    print(f"\n[INFO] Training sign model: {len(xt)} train / {len(xv)} validation samples, "
          f"{len(classes)} classes")

    net = SignNet(len(classes))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=(len(xt) + batch_size - 1) // batch_size)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(xt))
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            if len(batch) < 2:
                continue
            optimizer.zero_grad()
            criterion(net(augment_poses(xt[batch])), yt[batch]).backward()
            optimizer.step()
            scheduler.step()

        if epoch == 1 or epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                acc = (net(xv).argmax(1) == yv).float().mean().item()
            print(f"   epoch {epoch:2d}/{epochs}  val_acc={acc:.1%}")

    net.eval()
    with torch.no_grad():
        pred = net(xv).argmax(1)
    print("[INFO] Accuracy per sign (held-out validation):")
    for i, name in enumerate(classes):
        mask = yv == i
        if mask.any():
            print(f"   {name:16s} {(pred[mask] == i).float().mean().item():6.1%}  ({int(mask.sum())} samples)")
    print(f"[INFO] Overall: {(pred == yv).float().mean().item():.1%}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "classes": classes}, SIGN_FILE)
    print(f"[INFO] Sign model saved: {SIGN_FILE}")
    return classes


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    if os.path.exists(DATASET_DIR):
        people = train_lbph(DATASET_DIR)
        if people:
            train_cnn(DATASET_DIR)
    else:
        print("[INFO] No face data yet - run face_taker.py to enroll people.")

    train_signs()

    print("\n[INFO] Training finished.")
