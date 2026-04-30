"""Synthetic rare-disease data demo for Milestone 1.

Final project will use a diffusion model. This milestone script demonstrates
where synthetic rare-disease generation fits into the pipeline by creating
additional augmented images for the rare class.
"""

from pathlib import Path
import json
from PIL import Image, ImageEnhance, ImageOps

SOURCE_DIR = Path("data/sample_crop_dataset/leaf_spot")
TARGET_DIR = Path("data/sample_crop_dataset_synthetic/leaf_spot")
RESULTS_DIR = Path("results")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    generated = 0
    for img_path in list(SOURCE_DIR.glob("*.png"))[:10]:
        img = Image.open(img_path).convert("RGB")
        aug = ImageOps.mirror(img)
        aug = ImageEnhance.Color(aug).enhance(1.3)
        aug = ImageEnhance.Contrast(aug).enhance(1.2)
        out_path = TARGET_DIR / f"synthetic_{img_path.name}"
        aug.save(out_path)
        generated += 1

    summary = {
        "component": "synthetic rare disease data demo",
        "final_plan": "replace this augmentation demo with a diffusion-based generator",
        "synthetic_images_created": generated,
        "target_class": "leaf_spot",
        "output_dir": str(TARGET_DIR),
    }
    with open(RESULTS_DIR / "synthetic_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
