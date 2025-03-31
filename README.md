# Multi-Adapter Sentence Transformer Model for Retrieval-based QA

This project fine-tunes a SentenceTransformer model with multiple adapters for a retrieval-based QA system. The system uses a multi-adapter architecture, which allows the model to specialize in different domains for more efficient query-context retrieval.

## Installation

Ensure you have Python 3.7+ and install dependencies with:

```bash
pip install -r requirements.txt
```

## Requirements

```
torch>=1.10.0
transformers>=4.15.0
sentence-transformers>=2.2.0
datasets>=2.0.0
adapter-transformers>=3.0.0
tqdm>=4.62.0
numpy>=1.20.0
scikit-learn>=1.0.0
```

## Quick Start

```python
from model.adapters import MultiAdapterSentenceTransformer
from data.loader import QADataset

# Load the model
model = MultiAdapterSentenceTransformer(
    base_model="bert-base-uncased", 
    adapter_names=["medical", "legal", "technical"]
)

# Load a dataset
dataset = QADataset("path/to/dataset", domains=["medical", "legal", "technical"])

# Get embeddings for a query
query = "What are the symptoms of hypertension?"
embeddings = model.encode(query, adapter_name="medical")
```

## Training

To train the model, run:

```bash
python main.py --mode train --epochs 20 --batch_size 32 --lr 1e-7
```

## Evaluation

To evaluate the model, run:

```bash
python main.py --mode eval
```

## Model Architecture

The `MultiAdapterSentenceTransformer` class leverages adapter-based fine-tuning to create domain-specific embeddings:


## Directory Structure

```
viir_retrieval/
│
├── config.py           # All configuration values
├── main.py             # CLI entry point
├── train.py            # Training logic
├── evaluate.py         # Evaluation metrics and logic
├── data/
│   ├── loader.py       # Dataset loading, preprocessing
│   └── dataloader.py   # Custom data loaders
│
├── model/
│   ├── adapters.py     # Adapter model definition
│   └── loss.py         # Custom loss functions
│
├── utils/
│   ├── metrics.py      # Evaluation metrics
│   └── memory.py       # GPU memory cleanup
│
└── README.md
```

## Citation

If you use this code in your research, please cite:

```
@software{multi_adapter_sentence_transformer,
  author = {Your Name},
  title = {Multi-Adapter Sentence Transformer Model for Retrieval-based QA},
  year = {2025},
  url = {https://github.com/yourusername/viir_retrieval},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.