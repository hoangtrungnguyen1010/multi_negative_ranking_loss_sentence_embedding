import argparse
from config import Config
from data.loader import load_viir_dataset
from model import MultipleAdapterSentenceTransformer
from train import train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate multi-adapter retrieval model")
    parser.add_argument('--mode', choices=['train', 'eval'], required=True, help="train or eval mode")
    parser.add_argument('--epochs', type=int, default=Config.EPOCH)
    parser.add_argument('--batch_size', type=int, default=Config.BS)
    parser.add_argument('--lr', type=float, default=Config.LR)
    parser.add_argument('--output', type=str, default=Config.OUTPUT_PATH)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--AudreyTrungNguyen/Vi_IR', type=str, default=None)


    args = parser.parse_args()
    print(args)
    
    dataset = load_viir_dataset(args.model_name)

    # Load the model directly here
    model = MultipleAdapterSentenceTransformer(
        model_name_or_path=Config.BASE_MODEL_NAME,
        adapter_paths=Config.ADAPTER_PATHS,
        general_path=Config.GENERAL_ADAPTER_PATH
    )

    if args.mode == 'train':
        train_model(
            model=model,
            train_data=dataset['train'],
            val_data=dataset['validation'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_save_path=args.output
        )
    elif args.mode == 'eval':
        context_to_group = {item["context"]: item["group"] for item in dataset["test"]}

        # Extract unique contexts while preserving the order
        contexts = list(context_to_group.keys())

        # Get corresponding groups in the same order
        context_groups = [context_to_group[context] for context in contexts]
        
        list_of_docs = {'context': contexts, 'group': context_groups}

        # Assuming you have run_evaluation function implemented elsewhere
        if args.group != None:
            # Extract QA pairs
            questions = [item["question"] for item in dataset["test"] if item['group'] == args.group]
            ground_truth_contexts = [item["context"] for item in dataset["test"] if item['group'] == args.group]
            # Create a dictionary mapping context to group (if multiple groups exist for a context, it takes the first)
        else:
            questions = [item["question"] for item in dataset["test"]]
            ground_truth_contexts = [item["context"] for item in dataset["test"]]

            
        evaluate_model(questions, ground_truth_contexts, list_of_docs, dataset['test'])

if __name__ == "__main__":
    main()
