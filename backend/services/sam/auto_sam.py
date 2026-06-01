import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator

from backend.services.sam.sam_predictor import SAMPredictor


class AutoSAM:
    """Automatic segmentation using local Segment Anything ViT-H."""

    def __init__(self):
        self.sam = SAMPredictor()
        print("SAM ViT-H automatic mask generator ready")

    @staticmethod
    def _mask_iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / (union + 1e-6)

    @staticmethod
    def _contains_ratio(a, b):
        inter = np.logical_and(a, b).sum()
        area_a = a.sum()
        area_b = b.sum()
        return inter / (min(area_a, area_b) + 1e-6)

    @staticmethod
    def _clahe_rgb(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        out = cv2.merge((l2, a, b))
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

    @staticmethod
    def _aggressive_clahe(image):
        """Tight tile grid + high clip — maximises local contrast on metal surfaces."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l2 = clahe.apply(l)
        out = cv2.merge((l2, a, b))
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

    @staticmethod
    def _edge_enhanced(image):
        """
        Blend a Canny edge map into the luminance channel.
        Helps SAM find thin gasket boundaries and fine carburetor passages.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        # Dilate edges slightly so they are wide enough for SAM to pick up
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(image.astype(np.float32), 0.80,
                                  edge_rgb.astype(np.float32), 0.20, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _gamma_correct(image, gamma=1.5):
        """Brighten dark metallic recesses to expose hidden detail."""
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def _sharpen(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def _collect_masks(self, image, points_per_side, points_per_batch,
                       pred_iou_thresh, stability_score_thresh,
                       crop_n_layers, min_area_px):
        generator = SamAutomaticMaskGenerator(
            model=self.sam.model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_area_px,
        )

        try:
            outputs = generator.generate(image)
        except RuntimeError as e:
            print(f"[AutoSAM] automatic mask generation failed: {e}")
            return []
        except Exception as e:
            print(f"[AutoSAM] mask generation pass failed: {e}")
            return []

        return [item.get("segmentation") for item in outputs if "segmentation" in item]

    def _merge_masks(
        self,
        mask_arrays,
        min_area_px,
        max_area_ratio,
        dedup_iou,
        dedup_contains,
        max_masks,
    ):
        merged = []

        for m in mask_arrays:
            mask = np.array(m, dtype=bool)
            area = int(mask.sum())

            if area < min_area_px:
                continue

            if area > max_area_ratio * mask.size:
                continue

            duplicate = False
            for kept in merged:
                iou = self._mask_iou(mask, kept)
                contain = self._contains_ratio(mask, kept)
                if iou >= dedup_iou or contain >= dedup_contains:
                    duplicate = True
                    break

            if not duplicate:
                merged.append(mask)

            if len(merged) >= max_masks:
                break

        return merged

    def generate(self, image, config=None):
        """
        image: numpy uint8 array (H, W, 3)
        Returns list of dicts with 'segmentation' key (bool mask, same H x W).
        """
        config = config or {}

        recall_mode = config.get("recall_mode", "high")
        min_area_px = int(config.get("min_area_px", 30))
        max_masks = int(config.get("max_masks", 800))
        max_area_ratio = float(config.get("max_area_ratio", 0.95))
        dedup_iou = float(config.get("dedup_iou", 0.86))
        dedup_contains = float(config.get("dedup_contains", 0.97))

        params_by_mode = {
            "balanced": {
                "points_per_side": 24,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
                "crop_n_layers": 1,
                "dedup_iou": dedup_iou,
                "dedup_contains": dedup_contains,
            },
            "high": {
                "points_per_side": 32,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.84,
                "stability_score_thresh": 0.93,
                "crop_n_layers": 1,
                "dedup_iou": dedup_iou,
                "dedup_contains": dedup_contains,
            },
            "max": {
                "points_per_side": 40,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.80,
                "stability_score_thresh": 0.90,
                "crop_n_layers": 2,
                "dedup_iou": dedup_iou,
                "dedup_contains": dedup_contains,
            },
            # Tuned for complex metallic parts: gaskets (thin boundaries),
            # carburetors (many sub-components, nested passages).
            # Key changes vs "max":
            #   - points_per_side 64  → denser grid catches small passages & gasket edges
            #   - pred_iou_thresh 0.74 → accepts masks with uncertain boundaries
            #   - stability_score_thresh 0.82 → keeps less-stable thin-part masks
            #   - crop_n_layers 2      → hierarchical crops find sub-components
            #   - dedup_contains 0.88  → allows nested parts (gasket inside head)
            #   - dedup_iou 0.78       → treats similar-looking masks as duplicates only
            #                            when very similar (not just overlapping)
            "industrial": {
                "points_per_side": 64,
                "points_per_batch": 32,
                "pred_iou_thresh": 0.74,
                "stability_score_thresh": 0.82,
                "crop_n_layers": 2,
                "dedup_iou": 0.78,
                "dedup_contains": 0.88,
            },
        }
        mode_params = params_by_mode.get(recall_mode, params_by_mode["high"])

        # Override dedup from config if caller passed explicit values; otherwise
        # use the mode-specific defaults set above.
        effective_dedup_iou = float(config.get("dedup_iou", mode_params["dedup_iou"]))
        effective_dedup_contains = float(
            config.get("dedup_contains", mode_params["dedup_contains"])
        )

        variants = [image]
        if recall_mode in {"high", "max"}:
            variants.append(self._clahe_rgb(image))
        if recall_mode == "max":
            variants.append(self._sharpen(image))
        if recall_mode == "industrial":
            # Five variants: original, aggressive CLAHE, edge-enhanced,
            # gamma-brightened, and sharpened — covers the range of
            # lighting/contrast conditions on metal parts.
            variants.append(self._aggressive_clahe(image))
            variants.append(self._edge_enhanced(image))
            variants.append(self._gamma_correct(image, gamma=1.5))
            variants.append(self._sharpen(image))

        all_masks = []
        for variant in variants:
            all_masks.extend(
                self._collect_masks(
                    variant,
                    points_per_side=int(config.get("points_per_side", mode_params["points_per_side"])),
                    points_per_batch=int(config.get("points_per_batch", mode_params["points_per_batch"])),
                    pred_iou_thresh=float(config.get("pred_iou_thresh", mode_params["pred_iou_thresh"])),
                    stability_score_thresh=float(
                        config.get("stability_score_thresh", mode_params["stability_score_thresh"])
                    ),
                    crop_n_layers=int(config.get("crop_n_layers", mode_params["crop_n_layers"])),
                    min_area_px=min_area_px,
                )
            )

        merged = self._merge_masks(
            mask_arrays=all_masks,
            min_area_px=min_area_px,
            max_area_ratio=max_area_ratio,
            dedup_iou=effective_dedup_iou,
            dedup_contains=effective_dedup_contains,
            max_masks=max_masks,
        )

        return [{"segmentation": m} for m in merged]