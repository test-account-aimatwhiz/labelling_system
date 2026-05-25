import json
import os
import numpy as np


class CocoExporter:

    def __init__(self):

        self.data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

    def add_category(self, category_id, category_name):

        self.data["categories"].append({
            "id": category_id,
            "name": category_name
        })

    def add_image(
        self,
        image_id,
        file_name,
        width,
        height
    ):

        self.data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

    def add_annotation(
        self,
        annotation_id,
        image_id,
        category_id,
        segmentation,
        bbox,
        area
    ):

        self.data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })

    def save(self, output_path):

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=4)
