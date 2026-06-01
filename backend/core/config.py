import torch

class Settings:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_PATH = "backend/services/sam/checkpoints/sam_vit_h.pth"

    IMAGE_SIZE = 1024

    MASK_THRESHOLD = 0.5

    USE_FP16 = True

settings = Settings()
