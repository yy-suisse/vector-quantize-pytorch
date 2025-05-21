import torch
from vector_quantize_pytorch import ResidualVQ
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

# === Paths and constants ===
PATH = "D:/finetune_sbert_new/1Membeddings/"
EMBEDDING_PATH = PATH + "result_embeddings/sapbert_lora_triplet16.pt"
CONCEPT_PARQUET = PATH + "concept_all.parquet"
MAPPED_CSV = PATH + "mapped_concepts_2025-04-01.csv"
train_iter = 1000
alpha = 10
seed = 1234
lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)


# === Load data ===
df_concept_all = pl.read_parquet(CONCEPT_PARQUET)
df_mapped = pl.read_csv(MAPPED_CSV)
idx_mapped = df_mapped.join(df_concept_all, left_on="n.id", right_on="id")['idx'].unique().to_list()

# === Load and filter embeddings ===
embeddings = torch.load(EMBEDDING_PATH)['labels_embeddings']  # [N, 768]
embedding_mapped = embeddings[idx_mapped, :]  # [N', 768]

# === Dataset & DataLoader ===
dataset = TensorDataset(embedding_mapped)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# === Initialize RVQ model ===
model = ResidualVQ(
    dim=768,
    codebook_size=1000,
    num_quantizers=5,
    kmeans_init=True,
    kmeans_iters=10
).to(device)

# === Optimizer ===
opt = torch.optim.AdamW(model.parameters(), lr=lr)

# === Training loop ===
def train(model, train_loader, opt, train_iterations=1000, alpha=10):
    model.train()

    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x = next(data_iter)
            yield x[0].to(device)  # Extract the tensor from the tuple

    data_iter = iterate_dataset(train_loader)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x = next(data_iter)

        out, indices, cmt_loss = model(x)
        out = out.clamp(-1., 1.)  # optional, to stabilize training

        rec_loss = (out - x).abs().mean()
        total_loss = rec_loss + alpha * cmt_loss
        total_loss.backward()
        opt.step()

        pbar.set_description(
            f"rec: {rec_loss.item():.3f} | "
            f"cmt: {cmt_loss.item():.3f} | "
            f"total: {total_loss.item():.3f}"
        )

# === Run training ===
train(model, train_loader, opt, train_iterations=train_iter, alpha=alpha)