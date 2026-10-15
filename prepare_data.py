"""Extract Dataset1.zip into class folders; safe to rerun."""
from collections import Counter
from pathlib import Path
import zipfile
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "Dataset1.zip"
IMAGE_DIR = PROJECT_DIR / "data" / "images"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {"bus": "Bus", "car": "Car", "truck": "Truck", "motorcycle": "Motorcycle"}

def prepare_data():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {name: sum(1 for p in (IMAGE_DIR/name).glob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
              if (IMAGE_DIR/name).exists() else 0 for name in CLASS_NAMES.values()}
    if all(n > 0 for n in counts.values()):
        print("Using prepared images:", counts)
        return IMAGE_DIR
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset1.zip not found at {ZIP_PATH}. Copy your dataset ZIP beside train.py.")
    print("Extracting and checking dataset images...")
    copied, bad = Counter(), 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() not in VALID_EXTENSIONS:
                continue
            parts = [p.lower() for p in Path(member.filename.replace("\\\\", "/")).parts]
            class_key = next((key for key in CLASS_NAMES if key in parts), None)
            if class_key is None:
                # Kaggle exports sometimes have an extra wrapper folder; inspect path tokens.
                class_key = next((key for key in CLASS_NAMES if any(key == token.lower() for token in parts)), None)
            if class_key is None:
                continue
            try:
                with zf.open(member) as stream:
                    image = Image.open(stream).convert("RGB")
                    image.verify()
                with zf.open(member) as stream:
                    image = Image.open(stream).convert("RGB")
                    image.thumbnail((1024, 1024))
                    target_dir = IMAGE_DIR / CLASS_NAMES[class_key]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / f"{copied[CLASS_NAMES[class_key]]:04d}_{Path(member.filename).name}"
                    image.save(target, quality=92)
                    copied[CLASS_NAMES[class_key]] += 1
            except (UnidentifiedImageError, OSError, ValueError):
                bad += 1
    print("Prepared counts:", dict(copied), f"| skipped invalid images: {bad}")
    if any(copied[name] < 10 for name in CLASS_NAMES.values()):
        raise RuntimeError("Could not find enough images in all four classes. Check Dataset1.zip folder structure.")
    return IMAGE_DIR

if __name__ == "__main__":
    prepare_data()
