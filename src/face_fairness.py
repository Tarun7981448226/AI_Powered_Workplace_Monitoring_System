"""Step 5 (optional) - subgroup fairness and robustness study on FairFace.

FairFace (Karkkainen & Joo, CC BY 4.0, https://github.com/joojs/fairface) has
face crops labelled with race, gender and age. This script uses it as a testbed:
a CNN classifies gender, and accuracy is measured separately for each of the 7
race groups, on clean images and under noise, blur and a face-mask patch.

  1. baseline    - the CNN trained normally
  2. control     - the baseline trained for the same extra epochs, nothing else
  3. mitigated   - the baseline fine-tuned for those extra epochs with
       * context-aware synthetic data: extra noisy / blurred / masked copies,
         generated more often for the race groups and conditions where the
         baseline is weakest
       * adaptive loss reweighting: groups with higher error get a larger
         weight in the loss, re-estimated every epoch
       * selective fine-tuning: the first conv blocks stay frozen

Group errors for steps 3 are measured on a dev split of the training data,
never on the validation split used for the final numbers.

Output: data/fairness_report.json and a printed table. Run it with
    python src/face_fairness.py            (about 30-40 minutes on a laptop)
    python src/face_fairness.py --epochs 2 --subset 8000     (quick smoke test)

The dataset (about 560 MB) is downloaded once into dataset/fairface/.
"""
import argparse
import json
import os
import time
import urllib.request

import cv2
import numpy as np

from face_train import BASE_DIR, FaceCNN, add_face_mask, add_noise, get_device, nn, torch


# -----------------------------------------
# Paths and data
# -----------------------------------------
FAIRFACE_DIR = os.path.join(BASE_DIR, "dataset", "fairface")
CACHE_FILE = os.path.join(FAIRFACE_DIR, "fairface_64.npz")
REPORT_FILE = os.path.join(BASE_DIR, "data", "fairness_report.json")

HF_URL = "https://huggingface.co/datasets/HuggingFaceM4/FairFace/resolve/main/0.25/"
PARQUET_FILES = {
    "train": ["train-00000-of-00002-d405faba4f4b9b85.parquet",
              "train-00001-of-00002-dd3cb68164727418.parquet"],
    "val": ["validation-00000-of-00001-951dbd63c8724ee1.parquet"],
}
RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern",
         "Latino_Hispanic", "Southeast Asian"]
CONDITIONS = ["clean", "noise", "blur", "mask"]
SIZE = 64


def download_fairface():
    os.makedirs(FAIRFACE_DIR, exist_ok=True)
    for name in PARQUET_FILES["train"] + PARQUET_FILES["val"]:
        path = os.path.join(FAIRFACE_DIR, name)
        if not os.path.exists(path):
            print(f"[INFO] Downloading {name} ...")
            urllib.request.urlretrieve(HF_URL + name, path)


def read_split(names):
    import pyarrow.parquet as pq

    images, gender, race = [], [], []
    for name in names:
        table = pq.read_table(os.path.join(FAIRFACE_DIR, name), columns=["image", "gender", "race"])
        for img, g, r in zip(table.column("image").to_pylist(),
                             table.column("gender").to_pylist(),
                             table.column("race").to_pylist()):
            gray = cv2.imdecode(np.frombuffer(img["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
            images.append(cv2.equalizeHist(cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)))
            gender.append(g)
            race.append(r)
    return np.array(images, np.uint8), np.array(gender), np.array(race)


def load_fairface():
    """Grayscale 64x64 crops, cached as one .npz after the first run."""
    if not os.path.exists(CACHE_FILE):
        download_fairface()
        print("[INFO] Decoding FairFace (one time) ...")
        tx, tg, tr = read_split(PARQUET_FILES["train"])
        vx, vg, vr = read_split(PARQUET_FILES["val"])
        np.savez_compressed(CACHE_FILE, tx=tx, tg=tg, tr=tr, vx=vx, vg=vg, vr=vr)
    d = np.load(CACHE_FILE)
    return {k: d[k] for k in d.files}


# -----------------------------------------
# Corruptions (training uses random strength, evaluation fixed strength)
# -----------------------------------------
def corrupt(img, condition, rng, fixed=False):
    if condition == "noise":
        return add_noise(img, 25 if fixed else rng.uniform(10, 40), rng)
    if condition == "blur":
        k = 5 if fixed else int(rng.choice([3, 5, 7]))
        return cv2.GaussianBlur(img, (k, k), 0)
    if condition == "mask":
        return add_face_mask(img)
    return img


def corrupt_all(images, condition, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([corrupt(img, condition, rng, fixed=True) for img in images])


# -----------------------------------------
# Training and evaluation
# -----------------------------------------
def to_tensor(x, device, flip=False):
    t = (torch.from_numpy(x).float().unsqueeze(1) / 255 - 0.5) / 0.5
    if flip:
        mask = torch.rand(len(t)) < 0.5
        t[mask] = t[mask].flip(3)
        t = t * (1 + 0.2 * (torch.rand(len(t), 1, 1, 1) - 0.5)) + 0.2 * (torch.rand(len(t), 1, 1, 1) - 0.5)
    return t.to(device)


def predict(model, images, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            out.append(model(to_tensor(images[i:i + batch_size], device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def group_accuracy(model, images, gender, race, device):
    """Accuracy per race group under every condition."""
    result = {}
    for cond in CONDITIONS:
        x = images if cond == "clean" else corrupt_all(images, cond)
        hit = predict(model, x, device) == gender
        per_race = [float(hit[race == r].mean()) for r in range(len(RACES))]
        result[cond] = {"overall": float(hit.mean()), "per_race": per_race}
    return result


def group_errors(acc):
    """(race x condition) error grid from a group_accuracy() result."""
    return np.array([[1 - acc[c]["per_race"][r] for c in CONDITIONS[1:]] for r in range(len(RACES))])


def fit(model, data, epochs, lr, device, rng, freeze=0, dev=None, mitigate=None):
    """Train for `epochs`. `mitigate` = (error_grid) turns on context-aware
    synthetic data and adaptive loss reweighting."""
    x, y, race = data["x"], data["y"], data["race"]
    for i, block in enumerate(model.features[:-1]):
        for p in block.parameters():
            p.requires_grad = i >= freeze
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = (len(x) + 127) // 128
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    criterion = nn.CrossEntropyLoss(reduction="none")

    group_weight = np.ones(len(RACES))
    if mitigate is not None:
        err = mitigate
        difficulty = err.mean(1) / err.mean()
        synth_prob = np.clip(0.3 * difficulty, 0.05, 0.8)             # per race
        cond_prob = (err + 1e-3) / (err + 1e-3).sum(1, keepdims=True)  # per race over noise/blur/mask

    for epoch in range(1, epochs + 1):
        model.train()
        for block in model.features[:freeze]:
            block.eval()  # frozen blocks keep their BatchNorm statistics
        perm = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            batch = x[idx].copy()
            if mitigate is not None:
                for j, k in enumerate(idx):
                    r = race[k]
                    if rng.random() < synth_prob[r]:
                        batch[j] = corrupt(batch[j], CONDITIONS[1:][rng.choice(3, p=cond_prob[r])], rng)
            xb = to_tensor(batch, device, flip=True)
            yb = torch.from_numpy(y[idx]).to(device)
            w = torch.from_numpy(group_weight[race[idx]]).float().to(device)
            optimizer.zero_grad()
            loss = (criterion(model(xb), yb) * w).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)

        if mitigate is not None and dev is not None:  # adaptive loss reweighting
            hit = predict(model, dev["x"], device) == dev["y"]
            e = np.array([1 - hit[dev["race"] == r].mean() for r in range(len(RACES))]) + 1e-3
            group_weight = np.clip(e / e.mean(), 0.5, 2.0)
        print(f"      epoch {epoch}/{epochs}  loss={total / len(x):.4f}", flush=True)


def spread(acc):
    """Variance, worst-group accuracy and gap of the per-race accuracies."""
    out = {}
    for cond in CONDITIONS:
        a = np.array(acc[cond]["per_race"])
        out[cond] = {"variance": float(a.var()), "worst": float(a.min()), "gap": float(a.max() - a.min())}
    out["mean_variance"] = float(np.mean([out[c]["variance"] for c in CONDITIONS]))
    out["mean_worst"] = float(np.mean([out[c]["worst"] for c in CONDITIONS]))
    return out


def run_seed(seed, data, epochs, extra_epochs, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # dev split from the training data, used only to estimate group errors
    order = rng.permutation(len(data["tx"]))
    n_dev = int(0.08 * len(order))
    dev_i, tr_i = order[:n_dev], order[n_dev:]
    train = {"x": data["tx"][tr_i], "y": data["tg"][tr_i], "race": data["tr"][tr_i]}
    dev = {"x": data["tx"][dev_i], "y": data["tg"][dev_i], "race": data["tr"][dev_i]}

    print(f"\n[SEED {seed}] baseline: {epochs} epochs on {len(train['x'])} images", flush=True)
    base = FaceCNN(2, dropout=0.3).to(device)
    fit(base, train, epochs, 2e-3, device, rng)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    base_acc = group_accuracy(base, data["vx"], data["vg"], data["vr"], device)

    dev_acc = group_accuracy(base, dev["x"], dev["y"], dev["race"], device)  # errors for step 3
    err = group_errors(dev_acc)

    print(f"[SEED {seed}] control: +{extra_epochs} plain epochs", flush=True)
    control = FaceCNN(2, dropout=0.3).to(device)
    control.load_state_dict(base_state)
    fit(control, train, extra_epochs, 5e-4, device, rng)
    control_acc = group_accuracy(control, data["vx"], data["vg"], data["vr"], device)

    print(f"[SEED {seed}] mitigated: +{extra_epochs} epochs, synthetic data + reweighting + frozen blocks", flush=True)
    mitigated = FaceCNN(2, dropout=0.3).to(device)
    mitigated.load_state_dict(base_state)
    fit(mitigated, train, extra_epochs, 5e-4, device, rng, freeze=2, dev=dev, mitigate=err)
    mitigated_acc = group_accuracy(mitigated, data["vx"], data["vg"], data["vr"], device)

    return {"baseline": base_acc, "control": control_acc, "mitigated": mitigated_acc,
            "spread": {"baseline": spread(base_acc), "control": spread(control_acc),
                       "mitigated": spread(mitigated_acc)}}


def print_summary(runs):
    names = ["baseline", "control", "mitigated"]
    print("\n[RESULT] Gender classification on FairFace validation, accuracy per race group")
    print("         mean over seeds; spread = variance of the 7 per-race accuracies (x1e-4)")
    for cond in CONDITIONS:
        print(f"\n   {cond}")
        print(f"   {'':16s}" + "".join(f"{n:>12s}" for n in names))
        for r, race in enumerate(RACES):
            row = [np.mean([run[n][cond]["per_race"][r] for run in runs]) for n in names]
            print(f"   {race:16s}" + "".join(f"{v:12.1%}" for v in row))
        print(f"   {'spread (x1e-4)':16s}" + "".join(
            f"{np.mean([run['spread'][n][cond]['variance'] for run in runs]) * 1e4:12.2f}" for n in names))
        print(f"   {'worst group':16s}" + "".join(
            f"{np.mean([run['spread'][n][cond]['worst'] for run in runs]):12.1%}" for n in names))

    var = {n: np.mean([run["spread"][n]["mean_variance"] for run in runs]) for n in names}
    worst = {n: np.mean([run["spread"][n]["mean_worst"] for run in runs]) for n in names}
    print("\n   Across all four conditions:")
    for n in names:
        print(f"   {n:10s} mean variance {var[n] * 1e4:7.2f}e-4   mean worst-group accuracy {worst[n]:.1%}")
    for ref in ("baseline", "control"):
        print(f"   variance change of mitigated vs {ref}: {(var['mitigated'] - var[ref]) / var[ref]:+.1%}")
    return var, worst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--extra-epochs", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--subset", type=int, default=0, help="use only this many training images (smoke test)")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required for this script.")

    data = load_fairface()
    if args.subset:
        data["tx"], data["tg"], data["tr"] = (data[k][:args.subset] for k in ("tx", "tg", "tr"))
        data["vx"], data["vg"], data["vr"] = (data[k][:args.subset // 4] for k in ("vx", "vg", "vr"))
    device = get_device()
    print(f"[INFO] FairFace: {len(data['tx'])} train / {len(data['vx'])} validation images, device={device}")

    started = time.time()
    runs = [run_seed(seed, data, args.epochs, args.extra_epochs, device) for seed in range(args.seeds)]
    var, worst = print_summary(runs)

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump({"races": RACES, "conditions": CONDITIONS, "seeds": args.seeds, "epochs": args.epochs,
                   "extra_epochs": args.extra_epochs, "mean_variance": var, "mean_worst_group": worst,
                   "runs": runs}, f, indent=2)
    print(f"\n[INFO] Report saved: {REPORT_FILE}  ({(time.time() - started) / 60:.1f} min)")
