import numpy as np
# Function to compute Accuracy@K
def accuracy_at_k(retrieved_indices, ground_truth_indices, k):
    correct_retrievals = sum(1 for i in range(len(ground_truth_indices)) if ground_truth_indices[i] in retrieved_indices[i][:k])
    return correct_retrievals / len(ground_truth_indices)

# Function to compute Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(retrieved_indices, ground_truth_indices):
    mrr = 0.0
    for i in range(len(ground_truth_indices)):
        for rank, idx in enumerate(retrieved_indices[i], start=1):
            if idx == ground_truth_indices[i]:  # Found the correct context
                mrr += 1.0 / rank
                break
    return mrr / len(ground_truth_indices)

# Function to compute Mean Average Precision (MAP)
def mean_average_precision(retrieved_indices, ground_truth_indices, k):
    avg_precision_sum = 0.0
    for i in range(len(ground_truth_indices)):
        correct_count = 0
        precision_sum = 0.0
        for rank, idx in enumerate(retrieved_indices[i][:k], start=1):
            if idx == ground_truth_indices[i]:  # Correct context found
                correct_count += 1
                precision_sum += correct_count / rank
        avg_precision_sum += precision_sum / (1 if correct_count == 0 else correct_count)
    return avg_precision_sum / len(ground_truth_indices)

# Function to compute Normalized Discounted Cumulative Gain (NDCG@K)
def ndcg_at_k(retrieved_indices, ground_truth_indices, k):
    def dcg(scores):
        return sum((score / np.log2(idx + 2)) for idx, score in enumerate(scores))

    ndcg_sum = 0.0
    for i in range(len(ground_truth_indices)):
        relevance = [1 if idx == ground_truth_indices[i] else 0 for idx in retrieved_indices[i][:k]]
        ideal_relevance = sorted(relevance, reverse=True)
        ndcg_sum += dcg(relevance) / (dcg(ideal_relevance) if dcg(ideal_relevance) > 0 else 1)
    
    return ndcg_sum / len(ground_truth_indices)
def mean_average_precision_at_k(retrieved_indices, ground_truth_indices, k=5):
    """Compute MAP@K: Mean Average Precision at top-K."""
    avg_precision_sum = 0.0
    for i in range(len(ground_truth_indices)):
        correct_count = 0
        precision_sum = 0.0
        for rank, idx in enumerate(retrieved_indices[i][:k], start=1):
            if idx == ground_truth_indices[i]:  # Correct context found
                correct_count += 1
                precision_sum += correct_count / rank
        avg_precision_sum += precision_sum / (1 if correct_count == 0 else correct_count)  # Avoid division by zero
    return avg_precision_sum / len(ground_truth_indices)
