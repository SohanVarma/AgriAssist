"""Create a small sample crop disease dataset for Milestone 1.

This keeps the project runnable during evaluation even if the full PlantVillage
or field dataset is not downloaded yet. Images are simple synthetic leaf-like
patterns for pipeline validation, not final research data.
"""

from pathlib import Path
import random
import csv
from PIL import Image, ImageDraw, ImageFilter

SEED = 42
random.seed(SEED)

BASE_DIR = Path("data/sample_crop_dataset")
CLASSES = {
    "healthy_leaf": (55, 145, 60),
    "leaf_blight": (130, 110, 45),
    "leaf_spot": (75, 125, 55),
}


def draw_leaf(class_name: str, color: tuple[int, int, int], index: int) -> Image.Image:
    img = Image.new("RGB", (96, 96), (235, 245, 230))
    draw = ImageDraw.Draw(img)

    # leaf body
    cx, cy = 48 + random.randint(-4, 4), 48 + random.randint(-4, 4)
    bbox = (cx - 28, cy - 38, cx + 28, cy + 38)
    draw.ellipse(bbox, fill=color)
    draw.line((cx, cy - 32, cx, cy + 32), fill=(30, 90, 35), width=2)

    # disease patterns
    if class_name == "leaf_spot":
        for _ in range(9):
            x, y = random.randint(25, 70), random.randint(20, 75)
            r = random.randint(2, 5)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(95, 45, 25))
    elif class_name == "leaf_blight":
        for _ in range(4):
            x1, y1 = random.randint(20, 55), random.randint(15, 70)
            x2, y2 = x1 + random.randint(15, 30), y1 + random.randint(5, 18)
            draw.polygon([(x1, y1), (x2, y1+8), (x2-5, y2), (x1-3, y2-6)], fill=(155, 120, 45))

    return img.filter(ImageFilter.SMOOTH)


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = BASE_DIR / "metadata.csv"
    rows = []

    for class_name, color in CLASSES.items():
        class_dir = BASE_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(40):
            img = draw_leaf(class_name, color, i)
            file_path = class_dir / f"{class_name}_{i:03d}.png"
            img.save(file_path)
            rows.append({"image_path": str(file_path), "label": class_name, "source": "milestone1_synthetic_sample"})

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} sample images at {BASE_DIR}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
