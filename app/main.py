from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.services.predictor import predict
from app.utils.image import preprocess_image
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Server is working!"}

@app.post("/predict")
async def predict_tanaman(
    tanaman: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        print(f"[INFO] Menerima prediksi untuk: {tanaman}, file: {file.filename}")
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        hasil, confidence = predict(tanaman, image_array)

        return {
            "tanaman": tanaman,
            "hasil": hasil,
            "confidence": confidence
        }
    except Exception as e:
        print(f"[ERROR] Prediksi gagal: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")
