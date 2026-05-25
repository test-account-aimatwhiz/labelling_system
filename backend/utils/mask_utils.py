import cv2
import numpy as np

def resize_mask(mask, shape):
    """
    Resize SAM mask to match image shape safely.
    
    mask: (H,W) binary mask
    shape: (height, width) target image shape
    """

    return cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST
    )