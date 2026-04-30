"""Milestone 1 data pipeline.

Loads image metadata, validates image files, creates train/validation splits,
and logs dataset statistics.
"""

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_DIR = Path("data/sample_crop_dataset")
METADATA_PATH = DATASET_DIR / "metadata.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError("Run `python src/01_create_sample_dataset.py` first.")

    df = pd.read_csv(METADATA_PATH)
    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    df = df[df["exists"]].drop(columns=["exists"])

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    train_df.to_csv(RESULTS_DIR / "train_metadata.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "val_metadata.csv", index=False)

    summary = {
        "seed": SEED,
        "total_images_loaded": int(len(df)),
        "train_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "classes": sorted(df["label"].unique().tolist()),
        "class_distribution": df["label"].value_counts().to_dict(),
        "status": "Data pipeline working and data loaded",
    }

    with open(RESULTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
