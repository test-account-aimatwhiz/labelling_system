import os

def save_yolo_label(image_name, bbox, label, save_dir="outputs/labels"):
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, image_name.replace(".jpg", ".txt"))

    x, y, w, h = bbox

    # YOLO format (normalized)
    # class_id x_center y_center width height

    # For now we map label → class_id manually
    class_map = {"nut": 0, "bolt": 1, "gear": 2}

    class_id = class_map.get(label, 0)

    x_center = x + w / 2
    y_center = y + h / 2

    # NOTE: you must normalize using image size later
    line = f"{class_id} {x_center} {y_center} {w} {h}\n"

    with open(file_path, "a") as f:
        f.write(line)