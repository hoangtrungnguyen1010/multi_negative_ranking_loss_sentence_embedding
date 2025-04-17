import argparse
import math
import torch

from config import Config
from data.loader import load_viir_dataset, prepare_for_training_with_hard_negatives
from model import MultipleAdapterSentenceTransformer
from train import train_model, evaluate_model

def adaptive_training(model, dataset, args, max_top_k=5, min_improvement=0.01, max_no_improve_rounds=2):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    top_k = args.top_k
    batch_size = args.batch_size
    eval_steps = args.eval_steps
    initial_epochs = args.epochs

    best_score = -float('inf')
    no_improve_rounds = 0

    while top_k <= max_top_k:
        # epochs = max(1, math.ceil(initial_epochs / (top_k + 1)))
        print(f"\n🔁 Training with top_k={top_k}, epochs={initial_epochs}, batch_size={batch_size}")
        print(args.is_query)
        dataset['train'] = prepare_for_training_with_hard_negatives(dataset['train'], model, top_k=top_k, group=args.group)
        dataset['validation'] = prepare_for_training_with_hard_negatives(dataset['validation'], model, top_k=top_k, group=args.group)

        model = train_model(
            model=model,
            train_data=dataset['train'],
            val_data=dataset['validation'],
            epochs=initial_epochs,
            batch_size=batch_size,
            learning_rate=args.lr,
            eval_steps=eval_steps,
            model_save_path=args.output,
            patience=args.patience,
            accumulation_steps=args.accumulation_steps,
            top_k=top_k,
            is_query = args.is_query
        )

        score = evaluate_model(
            [item["query"] for item in dataset["validation"]],
            [item["positive"] for item in dataset["validation"]],
            {
                'context': [item["positive"] for item in dataset["validation"]],
                'group': [item["positive_group"] for item in dataset["validation"]]
            },
            model,
            is_query = args.is_query
        )

        print(f"✅ Score at top_k={top_k}: {score:.4f}")

        if score - best_score > min_improvement:
            best_score = score
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= max_no_improve_rounds:
                print(f"🛑 Stopping early at top_k={top_k}, no improvement after {max_no_improve_rounds} rounds.")
                break
        batch_size = int(batch_size * (top_k + 2) // (3+ top_k))

        top_k += 1
        # batch_size = max(8, batch_size // 2)
        # eval_steps *= 2

    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate multi-adapter retrieval model")
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both', help="train or eval mode")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=Config.LR)
    parser.add_argument('--output', type=str, default='vi_neg0')
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='AudreyTrungNguyen/Vi_IR')
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=908)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--max_top_k', type=int, default=0)
    parser.add_argument('--is_query', type=bool, default=False)
    parser.add_argument('--BASE_MODEL_NAME', type=str, default="keepitreal/vietnamese-sbert")

    args = parser.parse_args()
    
    if args.max_top_k < args.top_k:
        args.max_top_k = args.top_k

    print(args)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = MultipleAdapterSentenceTransformer(
        model_name_or_path=args.BASE_MODEL_NAME,
    ).to(DEVICE)

    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=model.device)
        model.load_state_dict(checkpoint)

    dataset = load_viir_dataset(args.dataset)

    if args.mode in ['train', 'both']:
        model = adaptive_training(model, dataset, args, max_top_k=args.max_top_k)

    if args.mode in ['eval', 'both']:
        context_to_group = {item["positive"]: item["positive_group"] for item in dataset["test"]}
        contexts = list(context_to_group.keys())
        context_groups = [context_to_group[context] for context in contexts]

        list_of_docs = {'context': contexts, 'group': context_groups}

        if args.group:
            questions = [item["query"] for item in dataset["test"] if item['positive_group'] == args.group]
            ground_truth_contexts = [item["positive"] for item in dataset["test"] if item['positive_group'] == args.group]
        else:
            questions = [item["query"] for item in dataset["test"]]
            ground_truth_contexts = [item["positive"] for item in dataset["test"]]

        evaluate_model(questions, ground_truth_contexts, list_of_docs, model, is_query=args.is_query, batch_size = 16)

if __name__ == "__main__":
    main()