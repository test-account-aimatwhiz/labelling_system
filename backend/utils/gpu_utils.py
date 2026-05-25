import torch

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_gpu_info():
    if not torch.cuda.is_available():
        return "CUDA not available"

    return {
        "device": torch.cuda.get_device_name(0),
        "memory_allocated_mb": round(
            torch.cuda.memory_allocated(0) / 1024**2,
            2
        ),
        "memory_reserved_mb": round(
            torch.cuda.memory_reserved(0) / 1024**2,
            2
        )
    }
