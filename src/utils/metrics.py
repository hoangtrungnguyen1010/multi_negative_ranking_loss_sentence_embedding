import numpy as np

# Function to compute Accuracy@K
def accuracy_at_k(retrieved_texts, ground_truth_texts, k):
    correct_retrievals = sum(1 for i in range(len(ground_truth_texts)) if ground_truth_texts[i] in retrieved_texts[i][:k])
    return correct_retrievals / len(ground_truth_texts)

# Function to compute Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(retrieved_texts, ground_truth_texts):
    mrr = 0.0
    for i in range(len(ground_truth_texts)):
        for rank, text in enumerate(retrieved_texts[i], start=1):
            if text == ground_truth_texts[i]:  # Found the correct context
                mrr += 1.0 / rank
                break
    return mrr / len(ground_truth_texts)

# Function to compute Mean Average Precision (MAP)
def mean_average_precision(retrieved_texts, ground_truth_texts, k):
    avg_precision_sum = 0.0
    for i in range(len(ground_truth_texts)):
        correct_count = 0
        precision_sum = 0.0
        for rank, text in enumerate(retrieved_texts[i][:k], start=1):
            if text == ground_truth_texts[i]:  # Correct context found
                correct_count += 1
                precision_sum += correct_count / rank
        avg_precision_sum += precision_sum / (1 if correct_count == 0 else correct_count)  # Avoid division by zero
    return avg_precision_sum / len(ground_truth_texts)

# Function to compute Normalized Discounted Cumulative Gain (NDCG@K)
def ndcg_at_k(retrieved_texts, ground_truth_texts, k):
    def dcg(scores):
        return sum((score / np.log2(idx + 2)) for idx, score in enumerate(scores))

    ndcg_sum = 0.0
    for i in range(len(ground_truth_texts)):
        relevance = [1 if text == ground_truth_texts[i] else 0 for text in retrieved_texts[i][:k]]
        ideal_relevance = sorted(relevance, reverse=True)
        ndcg_sum += dcg(relevance) / (dcg(ideal_relevance) if dcg(ideal_relevance) > 0 else 1)
    
    return ndcg_sum / len(ground_truth_texts)

# Function to compute Mean Average Precision at K (MAP@K)
def mean_average_precision_at_k(retrieved_texts, ground_truth_texts, k=5):
    avg_precision_sum = 0.0
    for i in range(len(ground_truth_texts)):
        correct_count = 0
        precision_sum = 0.0
        for rank, text in enumerate(retrieved_texts[i][:k], start=1):
            if text == ground_truth_texts[i]:  # Correct context found
                correct_count += 1
                precision_sum += correct_count / rank
        avg_precision_sum += precision_sum / (1 if correct_count == 0 else correct_count)  # Avoid division by zero
    return avg_precision_sum / len(ground_truth_texts)
