from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image) / 255.0
    return arr

