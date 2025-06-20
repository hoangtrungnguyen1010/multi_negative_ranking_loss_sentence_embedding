import argparse
import math
import torch

from config import Config
from data.loader import load_viir_dataset, prepare_for_training_with_hard_negatives
from model import get_sentence_transformer
from train import train_model, evaluate_model
from transformers import AutoTokenizer

def adaptive_training(model, dataset, args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    top_k = args.top_k
    batch_size = args.batch_size
    eval_steps = args.eval_steps
    initial_epochs = args.epochs

    best_score = -float('inf')
    no_improve_rounds = 0

    while top_k <= args.max_top_k:
        # epochs = max(1, math.ceil(initial_epochs / (top_k + 1)))
        print(f"\n🔁 Training with top_k={top_k}, epochs={initial_epochs}, batch_size={batch_size}")
        print(args.is_query)
        train_dataset = prepare_for_training_with_hard_negatives(dataset['train'], model, top_k=top_k, batch_size=args.eval_batchsize)
        val_dataset = prepare_for_training_with_hard_negatives(dataset['validation'], model, top_k=top_k, batch_size=args.eval_batchsize)

        model = train_model(
            model=model,
            train_data=train_dataset,
            val_data=val_dataset,
            epochs=initial_epochs,
            batch_size=batch_size,
            learning_rate=args.lr,
            eval_steps=eval_steps,
            model_save_path='temp',
            patience=args.patience,
            accumulation_steps=args.accumulation_steps,
            top_k=top_k,
            is_query = args.is_query,
            max_step = args.max_step
        )

        score = evaluate_model(
            [item["query"] for item in val_dataset],
            [item["positive"] for item in val_dataset],
            {
                'context': [item["positive"] for item in val_dataset],
            },
            model,
            is_query = args.is_query
        )

        print(f"✅ Score at top_k={top_k}: {score:.4f}")

        if score - best_score > args.min_improvement:
            best_score = score
            no_improve_rounds = 0
            torch.save(model.state_dict(), args.output)
            batch_size = int(batch_size * (top_k + 2) // (2+ top_k + args.step))
            top_k += args.step

        else:
            no_improve_rounds += 1
            if no_improve_rounds >= args.max_no_improve_rounds:
                print(f"🛑 Stopping early at top_k={top_k}, no improvement after {args.max_no_improve_rounds} rounds.")
                break
            batch_size = int(batch_size * (top_k + 2) // (2 + top_k - args.step))
            top_k -= args.step
            checkpoint = torch.load(args.output, map_location=model.device)
            model.load_state_dict(checkpoint)
            initial_epochs += 1
        # batch_size = max(8, batch_size // 2)
        # eval_steps *= 2
    if args.load_best_model_at_the_end:
        checkpoint = torch.load(args.output, map_location=model.device)
        model.load_state_dict(checkpoint)

    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate multi-adapter retrieval model")
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both', help="train or eval mode")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=Config.LR)
    parser.add_argument('--output', type=str, default='vi_neg0')
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='AudreyTrungNguyen/Vi_IR')
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--max_top_k', type=int, default=0)
    parser.add_argument('--step', type=int, default=1)

    parser.add_argument('--is_query', type=bool, default=False)
    parser.add_argument('--BASE_MODEL_NAME', type=str, default="keepitreal/vietnamese-sbert")
    parser.add_argument('--load_best_model_at_the_end', type=bool, default=True)
    parser.add_argument('--min_improvement', type=float, default=0.005)
    parser.add_argument('--max_no_improve_rounds', type=int, default=2)
    parser.add_argument('--max_step', type=int, default=None)
    parser.add_argument('--eval_batchsize', type=int, default=32)

    args = parser.parse_args()
    
    if args.max_top_k < args.top_k:
        args.max_top_k = args.top_k

    print(args)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = get_sentence_transformer(
        model_name_or_path=args.BASE_MODEL_NAME,
        device=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(args.BASE_MODEL_NAME)

    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=model.device)
        model.load_state_dict(checkpoint)

    dataset = load_viir_dataset(args.dataset, tokenizer)

    if args.mode in ['train', 'both']:
        model = adaptive_training(model, dataset, args)

    if args.mode in ['eval', 'both']:
        questions = [item["query"] for item in dataset["test"]]
        ground_truth_contexts = [item["positive"] for item in dataset["test"]]

        contexts = list(set(ground_truth_contexts))

        list_of_docs = {'context': contexts}
        evaluate_model(questions, ground_truth_contexts, list_of_docs, model, is_query=args.is_query, batch_size = args.eval_batchsize)

if __name__ == "__main__":
    main()