import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, InputExample
from tqdm import tqdm
# Custom Multiple Negatives Ranking Loss
import faiss
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor, device
from data.dataloader import NoDuplicatesDataLoader
from utils.metrics import *
def _convert_to_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input `a` to a PyTorch tensor if it is not already a tensor.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input array or tensor.

    Returns:
        Tensor: The converted tensor.
    """
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a


def _convert_to_batch(a: Tensor) -> Tensor:
    """
    If the tensor `a` is 1-dimensional, it is unsqueezed to add a batch dimension.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with a batch dimension.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a

def _convert_to_batch_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input data to a tensor with a batch dimension.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input data to be converted.

    Returns:
        Tensor: The converted tensor with a batch dimension.
    """
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a

def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings (Tensor): The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


# Custom Multiple Negatives Ranking Loss
def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)
    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)

    return torch.mm(a_norm, b_norm.transpose(0, 1))

def multiple_negatives_ranking_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor = None, 
                                    scale: float = 20.0, similarity_fct=torch.nn.functional.cosine_similarity) -> torch.Tensor:
    """
    Computes the multiple negatives ranking loss for a batch of anchor-positive-negative triplets.
    
    Args:
        anchor (Tensor): Embeddings for anchor sentences. [batch_size, 768]
        positives (Tensor): Embeddings for positive sentences.[batch_size, 768]
        negatives (Tensor): List of embeddings for negative sentences.[batch_size, num_negatives, 768]
        similarity_fct (function): Similarity function (default: cosine similarity).
        scale (float): Scaling factor for similarity scores.
    
    Returns:
        loss (Tensor): Computed loss value.
    """
    batch_size = anchor.size(0)

    if negatives is not None:
        candidates = torch.cat((positive, negatives.view(-1, negatives.size(-1))), dim=0)
        positive = candidates.view(-1, candidates.size(-1))  # [batch_size*(num_negatives+1), 768]

    # Calculate similarity scores
    scores = cos_sim(anchor, positive) * scale  # [batch_size, batch_size*(num_negatives+1)]
    range_labels = torch.arange(0, scores.size(0), device=scores.device)
    return torch.nn.functional.cross_entropy(scores, range_labels)


# Training function with dataset handling
def train_model(
    model,
    train_data,
    val_data,
    patience,
    accumulation_steps,
    val_batch_size=32,
    eval_steps=None,
    batch_size=16,
    epochs=10,
    top_k = 5,
    learning_rate=2e-5,
    model_save_path="best_model.pth",
    device=None,
    is_query = False
):
    """
    Train and validate the model with gradient accumulation.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad == True], lr=learning_rate, weight_decay=1e-2)
    train_dataloader = NoDuplicatesDataLoader(train_data, batch_size=batch_size)
    val_dataloader = NoDuplicatesDataLoader(val_data, batch_size=val_batch_size)
    if not eval_steps:
        eval_steps = len(train_dataloader)

    best_loss = float("inf")
    early_stop_counter = 0
    training_loss = 0
    step = 0

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, (queries, positives, negatives, positive_groups, negative_groups) in enumerate(tqdm(train_dataloader, desc="🔄 Training", leave=True)):
            step += 1  # Count each batch as a step
            # Compute embeddings
            anchor_embeddings = model.forward(queries, is_query = is_query, batch_size=batch_size)
            positive_embeddings = model.forward(positives, batch_size=batch_size)

            if not negatives:
                loss = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings)
            else:
                negatives_embeddings = model.forward(negatives, batch_size=batch_size).view(len(queries), top_k, -1)
                loss = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings, negatives_embeddings)

            training_loss += loss.item()

            # Backpropagate
            loss.backward()
            
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            # Evaluation
            if step % eval_steps == 0:
                model.eval()
                val_loss = 0
                print(f"Training loss: {training_loss/eval_steps:.4f}")
                training_loss = 0
                
                with torch.no_grad():
                    print("Validating...")
                    for queries, positives, negatives, positive_groups, negative_groups in tqdm(val_dataloader, desc="🔄 Validation", leave=True):
                        anchor_embeddings = model.encode(queries, is_query = is_query, batch_size=val_batch_size, convert_to_tensor=True, show_progress_bar = False)
                        positive_embeddings = model.encode(positives, batch_size=val_batch_size, show_progress_bar = False)
                        
                        if not negatives:
                            loss = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings)
                        else:
                            negatives_embeddings = model.encode(negatives, batch_size=batch_size).view(len(queries), top_k, -1)
                            loss = multiple_negatives_ranking_loss(anchor_embeddings, positive_embeddings, negatives_embeddings)
                        
                        val_loss += loss.item()
                    
                avg_val_loss = val_loss / len(val_dataloader)
                print(f" Validation Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    early_stop_counter = 0
                    torch.save(model.state_dict(), model_save_path)
                    print("Best model saved!")
                else:
                    early_stop_counter += 1
                    print(f"No improvement, patience: {early_stop_counter}/{patience}")
                
                if early_stop_counter > patience:
                    print("Early Stopping Triggered! Stopping Training.")
                    return model
                
                model.train()

    return model


def evaluate_model(
    list_of_queries,
    list_of_groundtruth,
    list_of_docs,
    model,
    is_query = False,
    batch_size=128
):
    # Compute embeddings
    question_embeddings = model.encode(list_of_queries, is_query = is_query, batch_size=batch_size, convert_to_numpy=True)
    context_embeddings = model.encode(list_of_docs['context'], batch_size=batch_size, convert_to_numpy=True)

    # ✅ Normalize embeddings for cosine similarity search
    faiss.normalize_L2(context_embeddings)
    faiss.normalize_L2(question_embeddings)

    # Use FAISS (Facebook AI Similarity Search) for fast retrieval
    index = faiss.IndexFlatIP(context_embeddings.shape[1])  # Inner product (cosine similarity)
    index.add(context_embeddings)  # Add all context embeddings to the FAISS index

    # Evaluate Retrieval Performance
    top_k = 5
    retrieved_results = []

    for i, query_embedding in enumerate(question_embeddings):
        _, retrieved_indices = index.search(query_embedding.reshape(1, -1), top_k)
        retrieved_results.append([list_of_docs['context'][k] for k in retrieved_indices[0]])

    # Compute Metrics
    acc_k = accuracy_at_k(retrieved_results, list_of_groundtruth, top_k)
    mrr = mean_reciprocal_rank(retrieved_results, list_of_groundtruth)
    map_k = mean_average_precision(retrieved_results, list_of_groundtruth, top_k)
    ndcg_k = ndcg_at_k(retrieved_results, list_of_groundtruth, top_k)

    # Compute MAP@5
    map5 = mean_average_precision_at_k(retrieved_results, list_of_groundtruth, k=top_k)

    print(f"Mean Average Precision @5 (MAP@5): {map5:.4f}")
    # Print Results
    print(f"Top-{top_k} Accuracy: {acc_k * 100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Mean Average Precision (MAP@{top_k}): {map_k:.4f}")
    print(f"Normalized Discounted Cumulative Gain (NDCG@{top_k}): {ndcg_k:.4f}")
    return ndcg_k