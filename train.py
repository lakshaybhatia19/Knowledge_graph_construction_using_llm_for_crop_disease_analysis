import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class LeafDataset(Dataset):
    def __init__(self, dataframe, transform, class_to_idx):
        self.paths = dataframe["path"].tolist()
        self.labels = [class_to_idx[label] for label in dataframe["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(name, num_classes):
    if name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model

def evaluate_loss_accuracy(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
            loss = criterion(outputs.float(), labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def train_one_model(name, train_loader, val_loader, class_weights, num_classes,
                    output_dir, epochs, learning_rate, device):
    model = build_model(name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_accuracy = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"best_{name}.pt"

    for epoch in range(epochs):
        model.train()
        start = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
            loss = criterion(outputs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_loss, val_accuracy = evaluate_loss_accuracy(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(
            f"{name} | epoch {epoch + 1}/{epochs} | "
            f"val loss {val_loss:.4f} | val accuracy {val_accuracy:.4f} | {elapsed:.0f}s"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)

    print(f"{name} best validation accuracy: {best_accuracy:.4f}")
    return best_accuracy

def main():
    parser = argparse.ArgumentParser(description="Train plant disease image classifiers from scratch.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--models", nargs="+", default=["efficientnet_b0", "resnet50"],
                        choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    splits_dir = Path(args.splits_dir)
    train_df = pd.read_csv(splits_dir / "train_list.csv")
    val_df = pd.read_csv(splits_dir / "val_list.csv")
    classes = sorted(train_df["label"].unique())
    class_to_idx = {label: index for index, label in enumerate(classes)}

    for frame in (val_df,):
        unknown = set(frame["label"]) - set(classes)
        if unknown:
            raise ValueError(f"Validation contains unknown classes: {unknown}")

    sample_paths = train_df["path"].sample(min(3000, len(train_df)), random_state=args.seed)
    resize_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    pixel_count = 0
    for image_path in sample_paths:
        with Image.open(image_path) as image:
            tensor = resize_tensor(image.convert("RGB"))
        channel_sum += tensor.sum((1, 2))
        channel_sum_sq += (tensor ** 2).sum((1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]
    mean = channel_sum / pixel_count
    std = (channel_sum_sq / pixel_count - mean ** 2).clamp_min(1e-8).sqrt()
    print("Training-set mean:", mean.tolist())
    print("Training-set std:", std.tolist())

    normalize = T.Normalize(mean.tolist(), std.tolist())
    train_transform = T.Compose([
        T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomRotation(15),
        T.ColorJitter(0.1, 0.1), T.ToTensor(), normalize
    ])
    eval_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])

    train_loader = DataLoader(
        LeafDataset(train_df, train_transform, class_to_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        LeafDataset(val_df, eval_transform, class_to_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    counts = train_df["label"].map(class_to_idx).value_counts().sort_index().to_numpy()
    class_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    output_dir = Path(args.output_dir)
    (output_dir / "class_names.json").write_text(json.dumps(classes, indent=2))
    (output_dir / "normalization.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))
    results = {}
    for model_name in args.models:
        results[model_name] = train_one_model(
            model_name, train_loader, val_loader, class_weights, len(classes),
            output_dir, args.epochs, args.learning_rate, device
        )
    (output_dir / "validation_results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
