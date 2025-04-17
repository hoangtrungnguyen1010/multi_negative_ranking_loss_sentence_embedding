from torch.utils.data import DataLoader
from collections import defaultdict
import torch
from tqdm import tqdm
import math

import random
def collate_fn(batch):
    queries = [item.get('query', '') for item in batch]
    positives = [item.get('positive', '') for item in batch]
    
    negatives = [neg for item in batch for neg in item.get('negatives', [])]
    positive_groups = [item.get('positive_group', '') for item in batch]
    negative_groups = [ng for item in batch for ng in item.get('negative_groups', [])]

    return queries, positives, negatives, positive_groups, negative_groups

class NoDuplicatesDataLoader:
    def __init__(self, train_dataset, batch_size: int):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        
        Args:
            train_dataset (Dataset): PyTorch Dataset containing training examples
            batch_size (int): Number of samples per batch
        """
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = collate_fn
        
        # Convert dataset to list for easier manipulation
        self.train_examples = train_dataset
        
        # Create an index list for shuffling
        self.indices = list(range(len(self.train_examples)))
        random.shuffle(self.indices)

    def __iter__(self):
        """
        Iterator that yields batches with no duplicate texts
        """
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                # Use index to get the current example
                if self.data_pointer >= len(self.indices):
                    # Reset and reshuffle when we reach the end
                    self.data_pointer = 0
                    random.shuffle(self.indices)

                # Get the current example using the shuffled index
                current_index = self.indices[self.data_pointer]
                example = self.train_examples[current_index]

                # Check for duplicate texts
                valid_example = True
                texts_to_check = []
                
                # Handle different possible text structures
                if isinstance(example, dict):
                    texts_to_check = [
                        example.get('query', ''),
                        example.get('positive', ''),
                    ] + example.get('negatives', [])
                elif hasattr(example, 'texts'):
                    texts_to_check = example.texts
                else:
                    # Fallback to string conversion
                    texts_to_check = [str(example)]

                # Normalize and check for duplicates
                for text in texts_to_check:
                    if not isinstance(text, str):
                        text = str(text)
                    
                    normalized_text = text.strip().lower()
                    if normalized_text in texts_in_batch:
                        valid_example = False
                        break

                # If no duplicates, add to batch
                if valid_example:
                    batch.append(example)
                    for text in texts_to_check:
                        if not isinstance(text, str):
                            text = str(text)
                        texts_in_batch.add(text.strip().lower())

                # Move to next index
                self.data_pointer += 1

            # Yield the batch using collate function if available
            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        """
        Calculate the number of batches
        """
        return math.floor(len(self.train_examples) / self.batch_size)
