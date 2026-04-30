"""Initial image classification model for Milestone 1.

This trains a small CNN on the sample crop disease dataset. The goal is not
final accuracy; it proves that the model pipeline runs and produces logged
preliminary results.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path("results")
TRAIN_CSV = RESULTS_DIR / "train_metadata.csv"
VAL_CSV = RESULTS_DIR / "val_metadata.csv"


class CropDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: dict[str, int]):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = label_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        x = self.transform(image)
        y = self.label_to_idx[row["label"]]
        return x, torch.tensor(y, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        raise FileNotFoundError("Run `python src/02_data_pipeline.py` first.")

    labels = sorted(pd.read_csv(TRAIN_CSV)["label"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(labels)}

    train_ds = CropDataset(TRAIN_CSV, label_to_idx)
    val_ds = CropDataset(VAL_CSV, label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = SmallCNN(num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    rows = []
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        rows.append({"epoch": epoch, "train_loss": avg_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_accuracy={val_acc:.3f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "training_metrics.csv", index=False)

    plt.figure()
    plt.plot(metrics["epoch"], metrics["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Milestone 1 Initial Model Loss")
    plt.savefig(RESULTS_DIR / "training_loss.png", bbox_inches="tight")

    torch.save({"model_state_dict": model.state_dict(), "label_to_idx": label_to_idx}, RESULTS_DIR / "initial_model.pt")
    print("Saved metrics, plot, and initial model to results/.")


if __name__ == "__main__":
    main()
