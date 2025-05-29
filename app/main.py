from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.services.predictor import predict
from app.utils.image import preprocess_image

app = FastAPI()

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
