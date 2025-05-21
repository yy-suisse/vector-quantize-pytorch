import os
import numpy as np
import torch
import torch.nn.functional as F
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
from vector_quantize_pytorch import ResidualVQ
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# === Config ===
class Config:
    path = "D:/finetune_sbert_new/1Membeddings/"
    embedding_path = path + "result_embeddings/sapbert_lora_triplet16.pt"
    concept_parquet = path + "concept_all.parquet"
    mapped_csv = path + "mapped_concepts_2025-04-01.csv"
    train_triplet = path + "training_anchor_idx_1M.parquet"
    model_save_path = "D:/finetune_sbert_new/1Membeddings/lora_16_quantized/"
    alpha = 0
    num_epochs = 200
    batch_size = 2048*2
    lr = 3e-4
    dim = 768
    codebook_size = 1000
    num_quantizers = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1234

torch.manual_seed(Config.seed)
# === Data Loading ===
def load_embeddings():
    df_concept_all = pl.read_parquet(Config.concept_parquet)
    df_mapped = pl.read_csv(Config.mapped_csv)
    idx_mapped = df_mapped.join(df_concept_all, left_on="n.id", right_on="id")["idx"].unique().to_list()
    full_embeddings_l = torch.load(Config.embedding_path)["labels_embeddings"]
    full_embeddings_exp = torch.load(Config.embedding_path)["expressions_embeddings"]
    mapped_embeddings = full_embeddings_l[idx_mapped, :]
    # full_embeddings = F.normalize(full_embeddings, dim=1)
    # mapped_embeddings = F.normalize(mapped_embeddings, dim=1)
    return full_embeddings_l, full_embeddings_exp, mapped_embeddings

def get_dataloader(embeddings):
    dataset = TensorDataset(embeddings)
    return DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

def initialize_model():
    model = ResidualVQ(
        dim=Config.dim,
        codebook_size=Config.codebook_size,
        num_quantizers=Config.num_quantizers,
        learnable_codebook=True,
        ema_update=False,
        use_cosine_sim=True
    ).to(Config.device)
    return model

def train(model, train_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    loss_list = []
    model.train()

    for _ in tqdm(range(Config.num_epochs)):
        for x_batch in train_loader:
            x = x_batch[0].to(Config.device)
            optimizer.zero_grad()

            out, _, _ = model(x)
            loss = 1 - F.cosine_similarity(x, out, dim=-1).mean()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if len(loss_list) % 100 == 0:
                print(f"Loss: {loss.item():.4f}")
    return model, loss_list

# === Evaluation ===
def evaluate(model, embeddings):
    model.eval()
    with torch.no_grad():
        embeddings = embeddings.to(Config.device)
        quantized, indices, _ = model(embeddings)
        cos_sim = F.cosine_similarity(embeddings, quantized, dim=-1).mean()
        print(f"Average cosine similarity: {cos_sim.item():.4f}")
        return quantized, indices

# === Visualization ===
def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Cosine Loss")
    plt.title("Training Loss")
    plt.show()

def plot_histograms(indices, num_quantizers=6):
    indices = indices.cpu().squeeze(0)
    for i in range(num_quantizers):
        plt.hist(indices[:, i], bins=30, histtype="step", label=f"Quantizer {i+1}")
    plt.legend(loc="upper right")
    plt.title("Token Usage per Quantizer")
    plt.show()

# === Save Model ===
def save_model(model):
    num_codes = Config.codebook_size
    num_quantizers = Config.num_quantizers
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'dim': Config.dim,
            'codebook_size': Config.codebook_size,
            'num_quantizers': Config.num_quantizers,
            'alpha': Config.alpha,
            'num_epochs': Config.num_epochs,
            'batch_size': Config.batch_size,
            'lr': Config.lr,
            'learnable_codebook': True,
            'ema_update': False,
            'use_cosine_sim': True

        }
    }, Config.model_save_path + 'quantized_lora_16_' + str(num_codes) + '_' + str(num_quantizers) + '.pt')