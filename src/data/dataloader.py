from torch.utils.data import DataLoader
from collections import defaultdict
import torch
from tqdm import tqdm
import random

class NoDuplicatesDataLoader:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_pointer = 0
        self.indices = list(range(len(self.dataset)))
        random.shuffle(self.indices)

    def __iter__(self):
        """
        Iterator for batches with no duplicate entries within a batch.
        """
        for _ in range(len(self) // self.batch_size):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                current_index = self.indices[self.data_pointer]
                example = self.dataset[current_index]

                texts_to_check = [
                    example.get('query', ''),
                    example.get('positive', ''),
                ] + example.get('negatives', [])
                
                if not any(text.strip().lower() in texts_in_batch for text in texts_to_check):
                    batch.append(example)
                    for text in texts_to_check:
                        texts_in_batch.add(text.strip().lower())
                
                self.data_pointer += 1
                if self.data_pointer >= len(self.indices):
                    self.data_pointer = 0
                    random.shuffle(self.indices)

            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
