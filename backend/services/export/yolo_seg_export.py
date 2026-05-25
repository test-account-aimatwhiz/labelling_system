import os
import hashlib
import time
import cv2
import numpy as np
from PIL import Image

CLASS_FILE = "dataset/classes.txt"
DATASET_YAML = "dataset/dataset.yaml"
IMAGES_ROOT = "dataset/images"
LABELS_ROOT = "dataset/labels"


def get_class_id(label):

    ensure_dataset_structure()
    label = label.strip()

    if not os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, "w") as f:
            pass

    with open(CLASS_FILE, "r") as f:
        raw = f.read()

    classes = [c.strip() for c in raw.splitlines() if c.strip()]

    if label not in classes:

        needs_newline_prefix = len(raw) > 0 and not raw.endswith("\n")

        with open(CLASS_FILE, "a") as f:
            if needs_newline_prefix:
                f.write("\n")
            f.write(label + "\n")

        classes.append(label)

    sync_dataset_yaml(classes)

    return classes.index(label)


def sync_dataset_yaml(classes):
    ensure_dataset_structure()

    quoted = ["'" + c.replace("'", "''") + "'" for c in classes]
    names_line = ", ".join(quoted)

    content = (
        "path: dataset\n\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(classes)}\n"
        f"names: [{names_line}]\n"
    )

    with open(DATASET_YAML, "w") as f:
        f.write(content)


def ensure_dataset_structure():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs(os.path.join(IMAGES_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(IMAGES_ROOT, "val"), exist_ok=True)
    os.makedirs(os.path.join(LABELS_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(LABELS_ROOT, "val"), exist_ok=True)


def initialize_dataset_metadata():
    ensure_dataset_structure()

    if not os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, "w") as f:
            pass

    with open(CLASS_FILE, "r") as f:
        classes = [c.strip() for c in f.readlines() if c.strip()]

    sync_dataset_yaml(classes)


def _choose_split(key, val_ratio=0.2):
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def save_training_image(image_name, image_np, val_ratio=0.2):
    ensure_dataset_structure()

    stem = os.path.splitext(os.path.basename(image_name))[0]
    fingerprint = hashlib.sha1(image_np.tobytes()).hexdigest()[:10]
    ts = int(time.time() * 1000)
    out_name = f"{stem}_{ts}_{fingerprint}.jpg"

    split = _choose_split(f"{stem}:{fingerprint}", val_ratio=val_ratio)
    save_dir = os.path.join(IMAGES_ROOT, split)
    out_path = os.path.join(save_dir, out_name)

    Image.fromarray(image_np).save(out_path, format="JPEG", quality=95)

    return out_name, split


def save_yolo_segmentation(
    image_name,
    image_shape,
    mask,
    label,
    split="train",
    labels_root=LABELS_ROOT
):

    ensure_dataset_structure()
    save_dir = os.path.join(labels_root, split)
    os.makedirs(save_dir, exist_ok=True)

    h, w = image_shape

    class_id = get_class_id(label)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return

    contour = max(contours, key=cv2.contourArea)

    points = contour.squeeze()

    if len(points.shape) != 2:
        return

    if points.shape[0] < 3:
        return

    normalized = []

    for p in points:

        x = float(np.clip(p[0] / w, 0.0, 1.0))
        y = float(np.clip(p[1] / h, 0.0, 1.0))

        normalized.append(f"{x:.6f}")
        normalized.append(f"{y:.6f}")

    line = f"{class_id} " + " ".join(normalized)

    txt_name = os.path.splitext(image_name)[0] + ".txt"

    path = os.path.join(save_dir, txt_name)

    with open(path, "a") as f:
        f.write(line + "\n")