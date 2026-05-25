from backend.services.sam.auto_sam import AutoSAM


class AnnotationPipeline:

    def __init__(self):
        # SAM3 via HuggingFace Transformers — downloads model on first run
        self.auto = AutoSAM()

    def run(self, image, config=None):
        masks = self.auto.generate(image, config=config)

        # always return list
        return masks if masks else []