import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Union, Optional
from collections import OrderedDict
from tqdm.autonotebook import trange
import numpy as np
from peft import LoraConfig, TaskType

class MultipleAdapterSentenceTransformer(nn.Module):
    def __init__(self, 
                model_name_or_path: str,
                general_adapter_path: Optional[str] = None,
                query_adapter_path: Optional[str] = None,
                device: Optional[str] = None,
                lora_config: Optional[Dict] = None):
        
        super(MultipleAdapterSentenceTransformer, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_adapter = None  # Track the current active adapter

        # Load the sentence transformer model
        self.sentence_transformer = SentenceTransformer(model_name_or_path).to(self.device)
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False  # Freeze base model parameters
        
        # Default LoRA configuration if not provided
        if lora_config is None:
            lora_config = {
                'r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.1,
                'task_type': TaskType.FEATURE_EXTRACTION
            }

        # Configure LoRA adapters
        lora_config_obj = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            task_type=lora_config['task_type']
        )

        # Add general adapter
        if general_adapter_path:
            self.sentence_transformer.add_adapter(lora_config_obj, adapter_name='general')
            self.sentence_transformer.load_adapter(general_adapter_path, 'general', is_trainable=True)
        else:
            self.sentence_transformer.add_adapter(lora_config_obj, adapter_name='general')

        # Add query adapter
        if query_adapter_path:
            self.sentence_transformer.add_adapter(lora_config_obj, adapter_name='question')
            self.sentence_transformer.load_adapter(query_adapter_path, 'question', is_trainable=True)
        else:
            self.sentence_transformer.add_adapter(lora_config_obj, adapter_name='question')
        
        # Set general adapter as default
        self.set_adapter('general')

    def set_adapter(self, adapter_name: str):
        """Set the active adapter and update tracking state"""
        if adapter_name not in ['general', 'question']:
            raise ValueError("adapter_name must be either 'general' or 'question'")
        
        self.sentence_transformer.set_adapter(adapter_name)
        self.current_adapter = adapter_name
        return self

    def set_adapter_for_training(self, adapter_name: str):
        """Set the adapter to be used during training and ensure only its parameters are trainable"""
        self.set_adapter(adapter_name)
        
        # Make only the current adapter trainable
        for name, param in self.sentence_transformer.named_parameters():
            if 'lora' in name:
                if adapter_name in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        return self

    def forward(self, sentences: List[str], is_query: bool = False,  
                batch_size: int = 32, convert_to_numpy: bool = False, 
                normalize_embeddings: bool = False, show_progress_bar: bool = False):
        """Process sentences using the appropriate adapter based on is_query."""
        device = self.device
        sentences = list(sentences)  # Ensure input is a list
        
        # Switch to appropriate adapter if needed
        adapter_to_use = 'question' if is_query else 'general'
        if self.current_adapter != adapter_to_use:
            self.set_adapter(adapter_to_use)
        # Tokenize and compute embeddings
        adapter_features = self.sentence_transformer.tokenize(sentences)
        adapter_features = {key: value.to(device) for key, value in adapter_features.items()}
        embeddings = self.sentence_transformer.forward(adapter_features)["sentence_embedding"]
        
        # Normalize embeddings if requested
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert embeddings to numpy if requested
        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        is_query: bool = False,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = True,
        device: str = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor], np.ndarray]:
        """
        Computes sentence embeddings efficiently with batch processing.
        """
        if isinstance(sentences, str):
            sentences = [sentences]  # Convert single sentence to list
        
        # Collect embeddings for each batch
        all_embeddings = []
        with torch.no_grad():
            for start_index in range(0, len(sentences), batch_size):
                batch = sentences[start_index : start_index + batch_size]
                embeddings = self.forward(
                    sentences=batch,
                    is_query=is_query,
                    batch_size=batch_size,
                    convert_to_numpy=False,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress_bar
                )
                all_embeddings.append(embeddings)
        
        # Concatenate results
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            if convert_to_numpy:
                all_embeddings = all_embeddings.cpu().numpy()
        
        return all_embeddings
    
    def save_adapters(self, general_adapter_path: str, query_adapter_path: str):
        """Save both adapters to specified paths"""
        # Save general adapter
        current_adapter = self.current_adapter
        
        self.set_adapter('general')
        self.sentence_transformer.save_adapter(general_adapter_path, 'general')
        
        self.set_adapter('question')
        self.sentence_transformer.save_adapter(query_adapter_path, 'question')
        
        # Restore original adapter state
        self.set_adapter(current_adapter)
        
        return self