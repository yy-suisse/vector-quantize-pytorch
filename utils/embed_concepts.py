import torch
import os

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import polars as pl
from tqdm.auto import tqdm

MODEL_CONFIG = {
    'finetune_1M': {
        'name': "yyzheng00/snomed_triplet_1M",
        'is_sentence_transformer': True,
        'needs_prefix': False,
        'use_cls_token': False
    },
    'baseline': {
        'name': "all-MiniLM-L6-v2",
        'is_sentence_transformer': True,
        'needs_prefix': False,
        'use_cls_token': False
    },
    "clinicalbert": {
        'name': "medicalai/ClinicalBERT",
        'is_sentence_transformer': False,
        'needs_prefix': False,
        'use_cls_token': False,
        'max_length': 256
    },
    "bio_clinicalbert": {
        'name': "emilyalsentzer/Bio_ClinicalBERT",
        'is_sentence_transformer': False,
        'needs_prefix': False,
        'use_cls_token': False,
        'max_length': 256
    },
    "e5_base": {
        'name': "intfloat/e5-base-v2",
        'is_sentence_transformer': True,
        'needs_prefix': True,
        'use_cls_token': False
    },
    "gte_base": {
        'name': "thenlper/gte-base",
        'is_sentence_transformer': True,
        'needs_prefix': True,
        'use_cls_token': False
    },
    "sapbert":{
        'name': "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        'is_sentence_transformer': False,
        'needs_prefix': False,
        'use_cls_token': True
    },
    "sapbert_lora_triplet8": {
        'name': "yyzheng00/sapbert_lora_triplet_rank8",
        'is_sentence_transformer': True,
        'needs_prefix': False,
        'use_cls_token': True
    },
    "sapbert_lora_triplet16": {
        'name': "yyzheng00/sapbert_lora_triplet_rank16_merged",
        'is_sentence_transformer': True,
        'needs_prefix': False,
        'use_cls_token': True
    }
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelWrapper:
    def __init__(self, model_name, device="cuda", is_sentence_transformer=True, 
                 needs_prefix=False, use_cls_token=False, max_length=None):
        self.device = torch.device(device)
        self.is_sentence_transformer = is_sentence_transformer
        self.needs_prefix = needs_prefix  # For E5/GTE models
        self.use_cls_token = use_cls_token  # For SapBERT-like models
        
        # Set default max_length based on model type if not provided
        self.max_length = max_length
        
        try:
            if is_sentence_transformer:
                self.model = SentenceTransformer(model_name, device=device)
                self.tokenizer = None
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                
                # Set default max_length based on model type if not provided
                if self.max_length is None:
                    if 'clinicalbert' in self.model.config._name_or_path.lower():
                        self.max_length = 256
                    else:
                        self.max_length = 512
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def encode(self, texts, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            print("Warning: Empty text list provided to encode()")
            return torch.zeros((0, self.model.get_sentence_embedding_dimension() 
                               if self.is_sentence_transformer else self.model.config.hidden_size))

        if self.needs_prefix:
            texts = [f"query: {t}" for t in texts]

        try:
            if self.is_sentence_transformer:
                # SentenceTransformer models already have tqdm inside encode()
                embeddings = self.model.encode(
                    texts, 
                    batch_size=batch_size, 
                    convert_to_tensor=True, 
                    device=self.device, 
                    show_progress_bar=True
                )
            else:
                # HuggingFace AutoModel case
                all_embeddings = []
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=True):
                    batch = texts[i:i+batch_size]
                    inputs = self.tokenizer(
                        batch, 
                        padding="max_length", 
                        max_length=self.max_length, 
                        truncation=True, 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        last_hidden = outputs.last_hidden_state

                        if self.use_cls_token:
                            batch_embeddings = last_hidden[:, 0, :]
                        else:
                            attention_mask = inputs['attention_mask'].unsqueeze(-1)
                            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
                            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                            batch_embeddings = sum_embeddings / sum_mask

                        all_embeddings.append(batch_embeddings.cpu())  # Move to CPU immediately to free GPU memory
                
                embeddings = torch.cat(all_embeddings, dim=0)
                # If we need the embeddings on GPU, we can move back with .to(self.device)
            
            return embeddings
            
        except Exception as e:
            print(f"Error during encoding: {e}")
            raise

def load_model(model_key, device):
    """Load model based on configuration dictionary"""
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model key: {model_key}")
    
    config = MODEL_CONFIG[model_key]
    
    return ModelWrapper(
        config['name'],
        device=device,
        is_sentence_transformer=config['is_sentence_transformer'],
        needs_prefix=config['needs_prefix'],
        use_cls_token=config['use_cls_token'],
        max_length=config.get('max_length')
    )

def save_embeddings(data_dict, file_path):
    """Helper function to save embeddings with error handling"""
    try:
        # Ensure all tensors are on CPU
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
                data_dict[key] = value.cpu()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        torch.save(data_dict, file_path)
        print(f"Successfully saved embeddings to {file_path}")
    except Exception as e:
        print(f"Error saving embeddings to {file_path}: {e}")