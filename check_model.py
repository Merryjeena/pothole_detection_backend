
from pathlib import Path
from ultralytics import YOLO

model_path = Path(__file__).resolve().parent / "best.pt"
model = YOLO(str(model_path))

print("Model task:", model.task)
print("Class names:", model.names)
print("Number of classes:", len(model.names))