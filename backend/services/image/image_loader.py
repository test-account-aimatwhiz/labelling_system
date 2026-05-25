import cv2
import numpy as np


class ImageLoader:

    @staticmethod
    def load_image(image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_rgb

    @staticmethod
    def get_image_size(image):

        height, width = image.shape[:2]

        return width, height
