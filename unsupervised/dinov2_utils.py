import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def embed_pil(self, image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        vec = outputs.last_hidden_state[:, 0]
        vec = torch.nn.functional.normalize(vec, dim=-1)
        return vec[0].detach().cpu().numpy()


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def crop_xyxy(image, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(image.width - 1, x1))
    y1 = max(0, min(image.height - 1, y1))
    x2 = max(x1 + 1, min(image.width, x2))
    y2 = max(y1 + 1, min(image.height, y2))
    return image.crop((x1, y1, x2, y2))

