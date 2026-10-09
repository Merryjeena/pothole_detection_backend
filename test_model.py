
from pathlib import Path
from ultralytics import YOLO
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent

YOLO_PATH = BASE_DIR / "best.pt"
AI_MODEL_PATH = BASE_DIR / "ai-image-detection"

if not YOLO_PATH.is_file():
    raise FileNotFoundError(f"YOLO model not found: {YOLO_PATH}")

if not AI_MODEL_PATH.is_dir():
    raise FileNotFoundError(f"AI model folder not found: {AI_MODEL_PATH}")

print("Loading YOLO pothole model...")
yolo_model = YOLO(str(YOLO_PATH))
print("YOLO model loaded successfully!")

print("Loading AI-image detector...")
ai_detector = pipeline(
    "image-classification",
    model=str(AI_MODEL_PATH),
    top_k=None,
)
print("AI-image detector loaded successfully!")

print("\nSUCCESS: Both models loaded!")