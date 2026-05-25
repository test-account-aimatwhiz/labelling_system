import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class SAMPredictor:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.model is intentionally public — AutoSAM uses it for
        # SamAutomaticMaskGenerator which requires the raw SAM model.
        self.model = sam_model_registry["vit_h"](
            checkpoint="checkpoints/sam_vit_h.pth"
        )

        self.model.to(device=device)
        self.predictor = SamPredictor(self.model)

        print("✅ SAM vit_h Loaded on", device)

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict_from_point(self, x, y):
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        best = np.argmax(scores)
        return masks[best].astype(np.uint8)