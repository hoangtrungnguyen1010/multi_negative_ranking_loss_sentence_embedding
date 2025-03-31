from torch.nn.utils.rnn import pad_sequence
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, 
    TrainingArguments, Trainer
)
from sentence_transformers import SentenceTransformer, losses
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import argparse


def prepare_dataset(dataset_name, tokenizer, batch_size, max_length, seed = 42, group = None):
    def collate_fn(batch):
        query_input_ids = [item["query_input_ids"] for item in batch]
        query_attention_masks = [item["query_attention_mask"] for item in batch]
        positive_input_ids = [item["positive_input_ids"] for item in batch]
        positive_attention_masks = [item["positive_attention_mask"] for item in batch]
        negative_input_ids = [item["negative_input_ids"] for item in batch]
        negative_attention_masks = [item["negative_attention_mask"] for item in batch]

        # Pad sequences to the longest sequence in the batch
        padded_query_input_ids = pad_sequence(query_input_ids, batch_first=True, padding_value=0)
        padded_query_attention_masks = pad_sequence(query_attention_masks, batch_first=True, padding_value=0)
        padded_positive_input_ids = pad_sequence(positive_input_ids, batch_first=True, padding_value=0)
        padded_positive_attention_masks = pad_sequence(positive_attention_masks, batch_first=True, padding_value=0)
        padded_negative_input_ids = pad_sequence(negative_input_ids, batch_first=True, padding_value=0)
        padded_negative_attention_masks = pad_sequence(negative_attention_masks, batch_first=True, padding_value=0)

        return {
            "query_input_ids": padded_query_input_ids,
            "query_attention_mask": padded_query_attention_masks,
            "positive_input_ids": padded_positive_input_ids,
            "positive_attention_mask": padded_positive_attention_masks,
            "negative_input_ids": padded_negative_input_ids,
            "negative_attention_mask": padded_negative_attention_masks,
        }
def train(args, train_dataset, val_dataset):
    """
    Trains a SentenceTransformer model with either full fine-tuning or LoRA.
    
    Args:
    - args: Training arguments (model name, training mode, hyperparameters).
    - train_dataset: Training dataset.
    - val_dataset: Validation dataset.
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Load model configuration
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Load model (Sentence Transformer)
    model = SentenceTransformer(args.model_name_or_path)

    # Apply LoRA if specified
    if args.method == "lora":
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        print("⚡ Using LoRA for fine-tuning")

    else:
        print("🔍 Full fine-tuning enabled")
    
    # Define Loss Function
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        push_to_hub=False,
        report_to=args.report_to,
        do_eval=args.do_eval
    )

    # Define evaluation function
    def compute_eval_loss(model, dataset):
        """Computes validation loss using MultipleNegativesRankingLoss."""
        eval_dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=False)
        loss_fn = losses.MultipleNegativesRankingLoss(model)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch_embeddings = model.encode(batch.texts, convert_to_tensor=True).to(device)
                loss = loss_fn(batch_embeddings)
                total_loss += loss.item()

        model.train()
        return total_loss / len(eval_dataloader)

    # Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_eval_loss
    )

    # Train Model
    trainer.train()

    # Save the trained model
    model.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model with full fine-tuning or LoRA.")
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help="Warmup ratio for LR scheduler")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per device")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for Adam optimizer")

    # Model and dataset parameters
    parser.add_argument('--model_name_or_path', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model checkpoint")
    parser.add_argument('--dataset_name', type=str, default="glue", help="Dataset to use")
    parser.add_argument('--output_dir', type=str, default="./output", help="Directory to save the model")
    parser.add_argument('--method', type=str, choices=["full", "lora"], default="full",
                        help="Training method: 'full' for full fine-tuning, 'lora' for parameter-efficient fine-tuning")

    args = parser.parse_args()

    # Run Training
    train(args)
