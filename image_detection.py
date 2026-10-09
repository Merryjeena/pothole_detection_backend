
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ai-image-detection"

snapshot_download(
    repo_id="capcheck/ai-image-detection",
    local_dir=str(MODEL_DIR),
)

print("\nAI image detector download complete.")
print(f"Model location: {MODEL_DIR}")