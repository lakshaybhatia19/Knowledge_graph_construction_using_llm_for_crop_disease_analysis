import argparse
import hashlib
import os
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_class_root(data_dir):
    data_dir = Path(data_dir)
    candidates = []
    for path in [data_dir, *data_dir.rglob("*")]:
        if path.is_dir():
            class_dirs = [p for p in path.iterdir() if p.is_dir()]
            if len(class_dirs) >= 2:
                image_count = sum(
                    1 for d in class_dirs for f in d.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                )
                if image_count:
                    candidates.append((image_count, path))
    if not candidates:
        raise RuntimeError("Could not find a folder containing class subfolders.")
    return max(candidates, key=lambda item: item[0])[1]

def extract_zip(zip_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    return find_class_root(destination)

def clean_dataset(data_dir):
    data_dir = Path(data_dir)
    seen = set()
    removed_duplicates = 0
    removed_unreadable = 0

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        clean_name = re.sub(r"_+", "_", class_dir.name).strip("_")
        if clean_name != class_dir.name:
            renamed = class_dir.with_name(clean_name)
            if renamed.exists():
                raise RuntimeError(f"Cannot rename {class_dir}: {renamed} already exists.")
            class_dir.rename(renamed)
            class_dir = renamed

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = hashlib.md5(image_path.read_bytes()).hexdigest()
            if digest in seen:
                image_path.unlink()
                removed_duplicates += 1
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                image_path.unlink()
                removed_unreadable += 1
                continue
            seen.add(digest)

    print(f"Duplicate images removed: {removed_duplicates}")
    print(f"Unreadable images removed: {removed_unreadable}")
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        count = sum(1 for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"{class_dir.name}: {count}")

def create_splits(data_dir, output_dir, seed=42):
    rows = []
    data_dir = Path(data_dir)
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": str(image_path.resolve()), "label": class_dir.name})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No supported image files found.")
    if df["label"].value_counts().min() < 3:
        raise RuntimeError("Each class needs at least 3 images for train/validation/test splitting.")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df.label, random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df.label, random_state=seed
    )

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for split_name, split_df in [
        ("train", train_df), ("val", val_df), ("test", test_df)
    ]:
        split_df.to_csv(output_dir / f"{split_name}_list.csv", index=False)
        for row in split_df.itertuples(index=False):
            target_dir = output_dir / split_name / row.label
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.path, target_dir / Path(row.path).name)
        print(f"{split_name}: {len(split_df)} images")

    print(f"Classes: {sorted(df.label.unique())}")
    print(f"Split files and image folders saved to: {output_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Clean and split a plant disease image dataset.")
    parser.add_argument("--zip", dest="zip_path", help="Path to the original dataset ZIP file.")
    parser.add_argument("--data-dir", help="Path to an already extracted class-folder dataset.")
    parser.add_argument("--work-dir", default="data", help="Working directory for extraction.")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory for splits.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if bool(args.zip_path) == bool(args.data_dir):
        parser.error("Provide exactly one of --zip or --data-dir.")

    data_dir = extract_zip(args.zip_path, args.work_dir) if args.zip_path else Path(args.data_dir)
    print(f"Dataset class directory: {data_dir.resolve()}")
    clean_dataset(data_dir)
    create_splits(data_dir, args.output_dir, args.seed)

if __name__ == "__main__":
    main()
