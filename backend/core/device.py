import torch


class DeviceManager:

    @staticmethod
    def get_device():
        """
        Automatically select best device
        """

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    @staticmethod
    def get_dtype():
        """
        Use fp16 on GPU for memory optimization
        """

        if torch.cuda.is_available():
            return torch.float16

        return torch.float32

    @staticmethod
    def gpu_info():

        if not torch.cuda.is_available():
            return {
                "device": "cpu"
            }

        return {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated":
                round(torch.cuda.memory_allocated(0) / 1024**3, 2),

            "memory_reserved":
                round(torch.cuda.memory_reserved(0) / 1024**3, 2),

            "total_memory":
                round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    2
                )
        }
