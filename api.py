
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
YOLO_PATH = BASE_DIR / "best.pt"
AI_MODEL_PATH = BASE_DIR / "ai-image-detection"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Road Damage Detection API",
    description="Pothole detection and image classification",
    version="1.0.0",
)

print("Loading YOLO model...")
yolo_model = YOLO(str(YOLO_PATH))

print("Loading AI-image detector...")
ai_detector = pipeline(
    "image-classification",
    model=str(AI_MODEL_PATH),
    top_k=None,
)

print("Both models are ready!")


@app.get("/")
def home():
    return {
        "message": "Road Damage Detection API is running",
        "yolo_loaded": yolo_model is not None,
        "ai_detector_loaded": ai_detector is not None,
    }


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WEBP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The file is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB.",
        )

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # 1. Detect potholes
    try:
        results = yolo_model.predict(
            source=image,
            conf=0.10,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0]) * 100

                detections.append({
                    "label": "pothole" if class_id == 0 else result.names[class_id],
                    "confidence": round(confidence, 2),
                    "box": [
                        round(float(v), 2)
                        for v in box.xyxy[0].tolist()
                    ],
                })

        pothole_detected = len(detections) > 0
        pothole_confidence = (
            max(d["confidence"] for d in detections)
            if detections else 0.0
        )

    except Exception as exc:
        print("YOLO inference error:", exc)
        raise HTTPException(
            status_code=500,
            detail="Pothole detection failed.",
        )

    # 2. Classify the image
    try:
        predictions = ai_detector(image)

        # Return raw predictions so we can verify the model's labels.
        # Do not assume label meanings before checking them.
        ai_predictions = [
            {
                "label": item["label"],
                "score": round(float(item["score"]) * 100, 2),
            }
            for item in predictions
        ]

    except Exception as exc:
        print("AI-image classification error:", exc)
        raise HTTPException(
            status_code=500,
            detail="Image classification failed.",
        )

    return {
        "success": True,
        "filename": file.filename,
        "pothole_detected": pothole_detected,
        "pothole_count": len(detections),
        "pothole_confidence": pothole_confidence,
        "detections": detections,
        "ai_image_predictions": ai_predictions,
        "message": (
            "Pothole detected. Review the results."
            if pothole_detected
            else "No pothole was detected."
        ),
    }