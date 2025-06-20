# Sentence Transformer Model for Retrieval-based QA

This project fine-tunes a [SentenceTransformer](https://www.sbert.net/) model for retrieval-based question answering (QA). The system is designed for efficient query-context retrieval using a standard sentence embedding model.

---

## Installation

Ensure you have Python 3.7+ and install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Quick Start

You can use the provided training and evaluation scripts, or use the model directly in your own code.

---

## Training

To train the model, run:

```bash
python src/main.py --mode train --epochs 20 --batch_size 32 --lr 1e-4 --dataset squad
```

Or use the provided shell script (runs in background and logs output):

```bash
bash run.sh
```

---

## Evaluation

To evaluate the model, run:

```bash
python src/main.py --mode eval --dataset squad
```

---

## Command Line Arguments

The main training/evaluation script (`src/main.py`) supports the following arguments:

- `--mode`: `train`, `eval`, or `both` (default: `both`)
- `--epochs`: Number of training epochs (default: 1)
- `--batch_size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: from config)
- `--output`: Output file for saving model checkpoints
- `--dataset`: Dataset name (default: `AudreyTrungNguyen/Vi_IR`)
- `--patience`: Early stopping patience
- `--accumulation_steps`: Gradient accumulation steps
- `--eval_steps`: Steps between evaluations
- `--top_k`: Number of hard negatives for training
- ...and more (see `src/main.py` for all options)

---

## Directory Structure

```
MixSentenceEmbedder/
│
├── src/
│   ├── config.py           # Configuration values
│   ├── main.py             # CLI entry point
│   ├── train.py            # Training and evaluation logic
│   ├── model.py            # SentenceTransformer loader
│   ├── data/
│   │   ├── loader.py       # Dataset loading, preprocessing
│   │   └── dataloader.py   # Custom data loaders
│   └── utils/
│       └── metrics.py      # Evaluation metrics
│
├── scripts/
│   ├── train.py            # (Optional) Additional training script
│   └── evaluate.py         # (Optional) Additional evaluation script
│
├── requirements.txt
├── run.sh
└── README.md
```

---

