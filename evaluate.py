import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

class LeafDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform):
        self.paths = dataframe["path"].tolist()
        self.labels = [class_to_idx[label] for label in dataframe["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]

def build_model(name, num_classes):
    if name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved plant disease classifiers.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    classes = json.loads((models_dir / "class_names.json").read_text())
    class_to_idx = {name: index for index, name in enumerate(classes)}
    dataframe = pd.read_csv(Path(args.splits_dir) / f"{args.split}_list.csv")
    normalization = json.loads((models_dir / "normalization.json").read_text())
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(normalization["mean"], normalization["std"])
    ])
    loader = DataLoader(
        LeafDataset(dataframe, class_to_idx, transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for model_name in ["efficientnet_b0", "resnet50"]:
        model_path = models_dir / f"best_{model_name}.pt"
        if not model_path.exists():
            print(f"Skipping {model_name}: checkpoint not found at {model_path}")
            continue
        model = build_model(model_name, len(classes)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        predictions, targets = [], []
        with torch.inference_mode():
            for images, labels in loader:
                outputs = model(images.to(device))
                predictions.extend(outputs.argmax(1).cpu().tolist())
                targets.extend(labels.tolist())

        accuracy = accuracy_score(targets, predictions)
        macro_f1 = f1_score(targets, predictions, average="macro")
        print(f"\n{model_name} | {args.split} accuracy: {accuracy:.4f} | macro F1: {macro_f1:.4f}")
        print(classification_report(
            targets, predictions, labels=list(range(len(classes))),
            target_names=classes, digits=3, zero_division=0
        ))
        results.append({
            "model": model_name, "split": args.split,
            "accuracy": float(accuracy), "macro_f1": float(macro_f1)
        })

        matrix = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
        plt.figure(figsize=(max(9, len(classes) * 0.55), max(8, len(classes) * 0.5)))
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(classes)), classes, rotation=90)
        plt.yticks(range(len(classes)), classes)
        plt.title(f"{model_name} - {args.split} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        Path("results").mkdir(exist_ok=True)
        plt.savefig(Path("results") / f"{model_name}_{args.split}_confusion_matrix.png", dpi=200)
        plt.close()

    if results:
        Path("results").mkdir(exist_ok=True)
        pd.DataFrame(results).to_csv(Path("results") / f"{args.split}_metrics.csv", index=False)

if __name__ == "__main__":
    main()
