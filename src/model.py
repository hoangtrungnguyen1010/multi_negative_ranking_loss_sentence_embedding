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
                adapter_paths: Optional[Dict[str, str]] = None,
                general_path: Optional[str] = None,
                classifier_path: Optional[str] = None,
                device: Optional[str] = None,
                lora_config: Optional[Dict] = None):
        
        super(MultipleAdapterSentenceTransformer, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        # Process adapter paths (can be None or empty for single adapter case)
        self.adapter_paths = adapter_paths or {}
        self.adapters = list(self.adapter_paths.keys())
        self.use_router = len(self.adapters) > 1  # Only use router if multiple adapters
        
        # Initialize router only if needed (multiple adapters)
        if self.use_router and len(self.adapters) > 0:
            embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
            num_classes = len(self.adapters)
            self.router = nn.Linear(embedding_dim, num_classes).to(self.device)
            
            # Load classifier if available and applicable
            if classifier_path:
                try:
                    checkpoint = torch.load(classifier_path, map_location=self.device)
                    model_state_dict = checkpoint.get("model_state_dict", {})
                    
                    new_state_dict = OrderedDict()
                    for key, value in model_state_dict.items():
                        new_key = key.replace("fc.", "")  # Remove "fc." to match model's expected keys
                        new_state_dict[new_key] = value
                    
                    self.router.load_state_dict(new_state_dict)
                    self.label_encoder = checkpoint.get("label_encoder", None)
                except Exception as e:
                    print(f"Warning: Could not load classifier from {classifier_path}: {e}")
        else:
            self.router = None
            self.label_encoder = None
        
        # Configure LoRA
        lora_config_obj = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            task_type=lora_config['task_type']
        )

        # Load general adapter if provided
        if general_path:
            self.sentence_transformer.add_adapter(lora_config_obj, 'general')
            try:
                self.sentence_transformer.load_adapter(general_path, 'general', is_trainable=True)
            except Exception as e:
                self.sentence_transformer.add_adapter(lora_config, 'general')
        
        # Load domain-specific adapters
        for adapter_name, path in self.adapter_paths.items():
            try:
                # Add adapter if it doesn't exist
                if adapter_name not in self.sentence_transformer.get_adapters():
                    self.sentence_transformer.add_adapter(lora_config_obj, adapter_name)
                
                self.sentence_transformer.load_adapter(path, adapter_name, is_trainable=True)
            except Exception as e:
                print(f"Warning: Could not load adapter {adapter_name} from {path}: {e}")
        

    def forward(self, sentences: List[str], batch_group: Union[list, np.ndarray, torch.Tensor] = None, 
                batch_size: int = 32, convert_to_numpy: bool = False, 
                normalize_embeddings: bool = False, show_progress_bar: bool = False):
        """Pass the input through the sentence transformer with routing logic."""
        device = self.device
        sentences = list(sentences)  # Ensure input is a list
        embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        
        # Simple case: Single adapter, no routing needed
        if not self.use_router:
            # Use the default adapter (or whatever is currently set)
            adapter_features = self.sentence_transformer.tokenize(sentences)
            adapter_features = {key: value.to(device) for key, value in adapter_features.items()}
            embeddings = self.sentence_transformer.forward(adapter_features)["sentence_embedding"]
            
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
                
            return embeddings
        
        # Multi-adapter case with routing logic
        
        # Get general embeddings for routing if no batch_group is provided
        if batch_group is None and self.has_general_adapter:
            self.sentence_transformer.set_adapter('general')
            
            adapter_features = self.sentence_transformer.tokenize(sentences)
            adapter_features = {key: value.to(device) for key, value in adapter_features.items()}
            general_embeddings = self.sentence_transformer.forward(adapter_features)["sentence_embedding"]
            
            # Pass embeddings through the router
            logits = self.router(general_embeddings)
            adapter_probs = torch.softmax(logits, dim=1)
            top_k_values, top_k_indices = torch.topk(adapter_probs, k=1, dim=1)
            
            # Create batch groups using label encoder
            batch_indices = top_k_indices.squeeze()
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                batch_group = self.label_encoder.inverse_transform(batch_indices.cpu().tolist())
            else:
                # Fall back to index-based mapping if no label encoder
                batch_group = [self.adapters[idx] for idx in batch_indices.cpu().tolist()]
                
            use_weighted_combo = True
        else:
            use_weighted_combo = False
            general_embeddings = None
            top_k_values = None
            
            # Process provided batch group
            if isinstance(batch_group, str):
                batch_group = [batch_group] * len(sentences)
            elif isinstance(batch_group, np.ndarray):
                batch_group = batch_group.tolist()
            elif isinstance(batch_group, torch.Tensor):
                batch_group = batch_group.cpu().tolist()
        
        # Process each unique route in batches
        domain_embeddings = torch.zeros((len(sentences), embedding_dim), device=device)
        unique_routes = list(set(batch_group))
        
        for route in unique_routes:
            # Set the adapter for this route
            if route in self.adapters:
                self.sentence_transformer.set_adapter(route)
            else:
                print(f"Warning: Adapter {route} not found, using default adapter")
                if self.default_adapter:
                    self.sentence_transformer.set_adapter(self.default_adapter)
                continue
            
            # Get indices for this route
            selected_indices = torch.tensor([idx for idx, grp in enumerate(batch_group) if grp == route], device=device)
            if selected_indices.numel() == 0:
                continue  # Skip if no sentences
            
            # Get sentences for this route
            selected_sentences = [sentences[idx] for idx in selected_indices.tolist()]
            
            # Process all at once
            adapter_features = self.sentence_transformer.tokenize(selected_sentences)
            adapter_features = {key: value.to(device) for key, value in adapter_features.items()}
            adapted_embeddings = self.sentence_transformer.forward(adapter_features)["sentence_embedding"]
            
            # Use scatter to preserve gradient flow
            domain_embeddings.scatter_(0, selected_indices.unsqueeze(1).expand(-1, embedding_dim), adapted_embeddings)
        
        # Apply weighted combination with general embeddings if auto-routing was used
        if use_weighted_combo and general_embeddings is not None and top_k_values is not None:
            res_embeddings = domain_embeddings * top_k_values + (1 - top_k_values) * general_embeddings
        else:
            res_embeddings = domain_embeddings
        
        # Normalize if requested
        if normalize_embeddings:
            res_embeddings = torch.nn.functional.normalize(res_embeddings, p=2, dim=1)
        
        # Convert to numpy if requested
        if convert_to_numpy:
            res_embeddings = res_embeddings.cpu().numpy()
        
        return res_embeddings
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        group: Union[int, str, List[Union[int, str]], np.ndarray, torch.Tensor] = None,
        batch_size: int = 32,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = True,
        device: str = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor], np.ndarray]:
        """
        Computes sentence embeddings efficiently with batch processing.
        """
    
        # Ensure sentences are in list format
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Convert `group` into tensor if necessary
        if group is not None:
            if isinstance(group, str):  
                group = [group]  # Convert single string to a list
            elif isinstance(group, np.ndarray):
                group = group.tolist()  # Convert NumPy array to a list
            elif isinstance(group, torch.Tensor):  
                group = group.cpu().tolist()  # Convert PyTorch tensor to a list
    
        if device is None:
            device = self.device
    
        self.to(device)
        self.eval()
        
        all_embeddings = []
    
        with torch.no_grad():
            for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
                batch = sentences[start_index : start_index + batch_size]
    
                # Handle batch_group properly
                if group is not None:
                    batch_group = group[start_index : start_index + len(batch)]
                else:
                    batch_group = None
    
                # Forward pass
                embeddings = self.forward(
                    sentences=batch,
                    batch_group=batch_group,
                    batch_size=batch_size,
                    convert_to_numpy=False,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=False
                )
    
                all_embeddings.append(embeddings)
    
        # Concatenate results and apply conversion if needed
        if len(all_embeddings) > 0:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            if convert_to_numpy:
                all_embeddings = all_embeddings.cpu().numpy()
        else:
            all_embeddings = np.array([]) if convert_to_numpy else torch.empty(0, device=device)
    
        return all_embeddings