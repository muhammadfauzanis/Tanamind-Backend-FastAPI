import tensorflow as tf
import numpy as np
import os

MODEL_DIR = "model"

# Map jenis tanaman ke file model dan label
MODEL_CONFIG = {
    "tomat": {
        "file": "tanamind_tomato.h5",
        "labels": [
            "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
            "Septoria_leaf_spot", "Target_Spot", "healthy", "powdery_mildew"
        ]
    },
    "cabai": {
        "file": "model_chili.h5",
        "labels": [
            "Whitefly", "Healthy2", "Anthracnose"
        ]
    },
    "selada": {
        "file": "lettuce_model.h5",
        "labels": [
            "Downy_mildew_on_lettuce", "Healthy", "Septoria_blight_on_lettuce", "Viral"
        ]
    }
}

# Cache model agar tidak di-load berulang
_loaded_models = {}

def get_model(tanaman: str):
    if tanaman not in MODEL_CONFIG:
        raise ValueError(f"Tanaman '{tanaman}' tidak didukung.")

    if tanaman not in _loaded_models:
        model_path = os.path.join(MODEL_DIR, MODEL_CONFIG[tanaman]["file"])
        model = tf.keras.models.load_model(model_path, compile=False)
        _loaded_models[tanaman] = model
        print(f"✅ Loaded model for {tanaman} from {model_path}")

    return _loaded_models[tanaman], MODEL_CONFIG[tanaman]["labels"]

def predict(tanaman: str, image_array: np.ndarray):
    model, labels = get_model(tanaman)
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    prediction = model.predict(image_array)
    pred_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Threshold check
    if confidence < 0.9:
        return "Unknown", confidence

    return labels[pred_class], confidence

