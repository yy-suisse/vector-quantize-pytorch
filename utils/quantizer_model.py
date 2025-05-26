import os
from vector_quantize_pytorch import ResidualVQ
import torch
import torch.nn.functional as F
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
from vector_quantize_pytorch import ResidualVQ
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np


# === Config ===
class Config:
    alpha = 0
    num_epochs = 1000
    batch_size = 2048*2
    lr = 3e-4
    dim = 768
    codebook_size = 512
    num_quantizers = np.arange(5, 10)  # Range of quantizers to test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1234
