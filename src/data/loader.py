from datasets import load_dataset, Dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_viir_dataset(model_name):
    """
    Loads the VIIR dataset from Hugging Face.
    This function returns the dataset as a dictionary.
    """
    dataset = load_dataset(model_name)
    
    # Renaming columns if they exist
    if 'question' in dataset['train'].column_names:
        dataset = dataset.rename_columns({'question': 'query'})
    if 'context' in dataset['train'].column_names:
        dataset = dataset.rename_columns({'context': 'positive'})
    if 'group' in dataset['train'].column_names:
        dataset = dataset.rename_columns({'group': 'positive_group'})

    # Accessing the different splits (train, validation, test)
    if model_name == 'squad':
        val_data = dataset['validation']
        test_data = val_data.train_test_split(test_size=0.5, seed=42)  # Random split
        val_data = test_data['train']
        test_data = test_data['test']
    else:
        val_data = dataset['validation']
        test_data = dataset['test']

    train_data = dataset['train']

    return {'train': train_data, 'validation': val_data, 'test': test_data}

def get_hard_negatives(question_embeddings,ground_contexts, contexts, groups, context_embeddings, num_negatives=5):
    """Retrieve hard negatives using cosine similarity (optimized batch processing)"""
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(question_embeddings, context_embeddings)
    # Sort to find most similar contexts (excluding correct context)
    sorted_indices = np.argsort(-similarity_matrix, axis=1)

    hard_negatives = []
    negative_groups = []
    for i in range(len(question_embeddings)):
        selected_idx = [idx for idx in sorted_indices[i] if contexts[idx] != ground_contexts[i]][:num_negatives]
        hard_negatives.append([contexts[idx] for idx in selected_idx])  # Exclude self
        negative_groups.append([groups[idx] for idx in selected_idx])
        
    return hard_negatives, negative_groups

def explode_negatives(data):
    """
    Explode the negatives and negative_groups to ensure they match in length
    
    Args:
        data (dict): Dictionary containing query, positive, negatives, etc.
    
    Returns:
        list: List of expanded negative entries
    """
    # Ensure negatives and negative_groups have the same length
    negatives = data['negatives']
    negative_groups = data['negative_groups']
    
    # If lengths don't match, raise an error
    if len(negatives) != len(negative_groups):
        raise ValueError(f"Negatives length ({len(negatives)}) does not match negative_groups length ({len(negative_groups)})")
    
    # Create a list of dictionaries with exploded negatives
    exploded_negatives = []
    for negative, group in zip(negatives, negative_groups):
        exploded_entry = {
            'query': data['query'],
            'positive': data['positive'],
            'negative': [negative],
            'negative_group': [group],
            'positive_group': data['positive_group']
        }
        exploded_negatives.append(exploded_entry)
    
    return exploded_negatives
# Example usage
def process_dataset(original_dataset):
    """
    Process the entire dataset to explode negatives
    
    Args:
        original_dataset (list): Original dataset with grouped negatives
    
    Returns:
        list: Processed dataset with exploded negatives
    """
    processed_dataset = []
    for data_point in original_dataset:
        # Explode the negatives for each data point
        exploded_entries = explode_negatives(data_point)
        processed_dataset.extend(exploded_entries)
    
    return processed_dataset


def prepare_for_training_with_hard_negatives(dataset, model, top_k = 5, batch_size= 128, group = None, is_exploded = None):
    if top_k <1:
        return dataset
    """prepare dataset for training, create hard sample if it's needed
    """
        # Extract QA pairs (Train + Validation)
    questions_train = [item["query"] for item in dataset if not group or item['positive_group'] == group]
    contexts_train = [item["positive"] for item in dataset if not group or item['positive_group'] == group]

    groups_train = [item["positive_group"] for item in dataset if not group or item['positive_group'] == group]
    
    if top_k == 0:
        train_data_dict = {
        "query": questions_train,
        "positive": contexts_train,
        "positive_group": groups_train,
    }
        
    else:
        context_to_group_train = {item["positive"]: item["positive_group"] for item in dataset}

        unique_contexts_train = list(context_to_group_train.keys())
        unique_group_train = [context_to_group_train[context] for context in unique_contexts_train]

        # ==========================
        # Encode Questions in Batches First (Faster)
        # ==========================
        question_embeddings_train = model.encode(questions_train, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

        # Encode contexts in Batches
        context_embeddings_train = model.encode(unique_contexts_train, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

        hard_negatives_train, negative_groups_train = get_hard_negatives(question_embeddings_train, contexts_train, unique_contexts_train, unique_group_train, context_embeddings_train, num_negatives=top_k)
        train_data_dict = {
            "query": questions_train,
            "positive": contexts_train,
            "negatives": hard_negatives_train, # List of negatives
            "positive_group": groups_train,
            "negative_groups": negative_groups_train

        }

    # ==========================
    # Convert to Hugging Face Dataset
    # ==========================
    train_dataset = Dataset.from_dict(train_data_dict)
    
    if top_k > 1 and is_exploded:
        train_dataset = process_dataset(train_dataset)
    return train_dataset
