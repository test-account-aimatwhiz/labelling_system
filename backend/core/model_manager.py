import os
import gc
import torch

from backend.core.device import DeviceManager
from backend.core.logger import logger
from backend.core.exceptions import (
    ModelLoadError,
    OutOfMemoryError
)


class ModelManager:

    def __init__(self):

        self.device = DeviceManager.get_device()
        self.dtype = DeviceManager.get_dtype()

        self.model = None
        self.checkpoint = None

        logger.info(f"Using device: {self.device}")

    def clear_memory(self):

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_checkpoint(self):

        checkpoint_path = "checkpoints/sam_vit_h.pth"

        if not os.path.exists(checkpoint_path):
            raise ModelLoadError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        logger.info("Loading SAM ViT-H checkpoint...")

        try:

            self.checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            logger.info("Checkpoint loaded successfully")

            return self.checkpoint

        except RuntimeError as e:

            if "out of memory" in str(e).lower():

                self.clear_memory()

                raise OutOfMemoryError(
                    "GPU memory exhausted while loading model"
                )

            raise ModelLoadError(str(e))

        except Exception as e:
            raise ModelLoadError(str(e))

    def inspect_checkpoint(self):

        if self.checkpoint is None:
            self.load_checkpoint()

        logger.info("Inspecting checkpoint structure")

        if isinstance(self.checkpoint, dict):

            print("\n===== CHECKPOINT KEYS =====\n")

            for key in self.checkpoint.keys():
                print(key)

            print("\n===========================\n")

            return list(self.checkpoint.keys())

        return []

    def checkpoint_summary(self):

        if self.checkpoint is None:
            self.load_checkpoint()

        summary = {
            "type": str(type(self.checkpoint)),
            "device": self.device
        }

        if isinstance(self.checkpoint, dict):
            summary["num_keys"] = len(self.checkpoint.keys())

        return summary
