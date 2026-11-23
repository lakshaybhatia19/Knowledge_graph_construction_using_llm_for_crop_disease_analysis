# Plant Disease Classification

An image-classification project using **EfficientNet-B0** and **ResNet-50**, trained from scratch with PyTorch. The workflow includes duplicate-image removal, image validation, stratified train/validation/test splitting, model training, and evaluation.

## Project structure

```text
plant-disease-classification/
├── prepare_dataset.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/       # created locally; do not upload the dataset to GitHub
├── models/     # trained checkpoints; do not upload large files
└── results/    # metrics and confusion-matrix plots
```

## Dataset format

The original dataset should contain one folder per class, with images inside each class folder. For example:

```text
dataset/
├── Tomato___Early_blight/
│   ├── image1.jpg
│   └── image2.jpg
├── Potato___Late_blight/
│   ├── image1.jpg
│   └── image2.jpg
└── ...
```

The preparation script can extract a ZIP file or use an already extracted dataset directory.

## Setup

Python 3.10 or later is recommended.

```bash
git clone <your-repository-url>
cd plant-disease-classification
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build suitable for your system from the official PyTorch installation selector if the default installation does not support your GPU.

## 1. Prepare the dataset

For a ZIP dataset:

```bash
python prepare_dataset.py --zip path/to/plantvillage.zip
```

For an already extracted dataset:

```bash
python prepare_dataset.py --data-dir path/to/dataset
```

The script:
- locates the class-folder directory;
- normalizes repeated underscores in class folder names;
- removes duplicate images using file hashes;
- removes unreadable images;
- creates stratified 70% training, 15% validation, and 15% test splits;
- saves split CSV files and split image folders under `data/splits/`.

The script prints the number of images found in each class and split.

## 2. Train the models

Train both models for 30 epochs:

```bash
python train.py --epochs 30
```

Train only EfficientNet-B0:

```bash
python train.py --models efficientnet_b0 --epochs 30
```

Train only ResNet-50:

```bash
python train.py --models resnet50 --epochs 30
```

The models are initialized with `weights=None`, meaning ImageNet pretrained weights are not used. The best checkpoint for each model is selected using validation accuracy and saved in `models/`.

## 3. Evaluate the models

Evaluate the saved checkpoints on the test split:

```bash
python evaluate.py --split test
```

For validation-set evaluation:

```bash
python evaluate.py --split val
```

Evaluation reports accuracy, macro F1-score, and per-class precision/recall/F1. Confusion-matrix images and a metrics CSV are saved under `results/`.

## Notes

- The dataset, model checkpoints, and generated results are excluded from Git by `.gitignore`. Keep large datasets and model files in an appropriate external storage location.
- The training script calculates channel mean and standard deviation from a sample of training images. The evaluation script currently uses only resizing and tensor conversion; for a strict, reproducible evaluation pipeline, use the same normalization values saved from training. 
- The code trains models from scratch. This can require substantial GPU time and a sufficiently large, balanced dataset.
- Results depend on dataset quality, class balance, hardware, and training settings. Do not interpret validation accuracy as final test performance.
