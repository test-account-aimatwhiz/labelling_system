import cv2
import numpy as np


class PolygonExtractor:

    @staticmethod
    def mask_to_polygon(mask):

        mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []

        for contour in contours:

            contour = contour.flatten().tolist()

            if len(contour) >= 6:
                polygons.append(contour)

        return polygons
