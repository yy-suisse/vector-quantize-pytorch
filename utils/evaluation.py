import ast
import numpy as np
import torch

def top_k_array_by_batch(id_concept_test_set, query_matrix, candidate_matrix, batch_size=100):
    ranks = []
    id_concept_test = list(id_concept_test_set)
    num_concepts = len(id_concept_test)
    num_batches = (num_concepts + batch_size - 1) // batch_size

    # Move tensors to memory-efficient float32 if not already
    query_matrix = torch.tensor(query_matrix, dtype=torch.float32)
    candidate_matrix = torch.tensor(candidate_matrix, dtype=torch.float32)

    candidate_matrix_T = candidate_matrix.T  # Precompute transpose

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_concepts)
        batch_indices = id_concept_test[start_idx:end_idx]
        print(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})")

        query_batch = query_matrix[batch_indices]

        # Compute scores
        scores = torch.matmul(query_batch, candidate_matrix_T)

        # Sort scores descending
        sorted_indices = torch.argsort(scores, dim=1, descending=True)

        # Find ranks
        for i, test_idx in enumerate(batch_indices):
            pos = (sorted_indices[i] == test_idx).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                rank = pos[0].item()
            else:
                rank = -1
            ranks.append(rank)

    return np.array(ranks)


def top_k_exp_by_batch(idx_true, query_matrix, candidate_matrix, device, batch_size=100):
    ranks = []

    num_concepts = len(idx_true)

    # Move tensors to memory-efficient float32 if not already
    query_matrix = torch.tensor(query_matrix, dtype=torch.float32).to(device)
    candidate_matrix = torch.tensor(candidate_matrix, dtype=torch.float32).to(device)

    candidate_matrix_T = candidate_matrix.T  # Precompute transpose

    for i in range(len(idx_true)):
        if i % batch_size == 0:
            print(f"Processing batch {i}/{num_concepts})")

        query = query_matrix[i]

        # Compute scores
        scores = torch.matmul(query, candidate_matrix_T)

        # Sort scores descending
        sorted_indices = torch.argsort(scores, descending=True)

        # Find ranks
        test_idx = idx_true[i]
        pos = (sorted_indices == test_idx).nonzero(as_tuple=True)

        rank = pos[0].item()

        ranks.append(rank)

    return np.array(ranks)


def top_k_array_syn(idx_syns_test, idx_syns, query_matrix, candidate_matrix, batch_size=512, device="cpu"):
    """
    idx_syns: List[int] → correct target indices
    query_matrix: Tensor [num_queries, dim]
    candidate_matrix: Tensor [num_candidates, dim]
    batch_size: number of queries to process at a time
    device: 'cuda' or 'cpu'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ranks = []

    # Prepare tensors and move to device
    query_matrix = torch.tensor(query_matrix, dtype=torch.float32).to(device)
    candidate_matrix = torch.tensor(candidate_matrix, dtype=torch.float32).to(device)
    candidate_matrix_T = candidate_matrix.T  # (dim, num_candidates)
    idx_syns = torch.tensor(idx_syns, device=device)
    idx_syns_test_set = set(idx_syns_test)

    for i, idx_syn in enumerate(idx_syns):
        if i % 1000 == 0:
            print(i, "/", len(idx_syns))
        if idx_syn.item() in idx_syns_test_set:
            query = query_matrix[i]  # (batch_size, dim)

            # Compute scores for the batch
            scores = torch.matmul(query, candidate_matrix_T)  # (batch_size, num_candidates)

            # Sort each query's candidates
            sorted_indices = torch.argsort(scores, descending=True)

            # Find the rank of the correct label
            matches = sorted_indices == idx_syn  # (batch_size, num_candidates)
            batch_ranks = matches.float().argmax()  # (batch_size)

            ranks.append(np.array([batch_ranks.cpu().numpy()]))  # move back to CPU
        else:
            continue

    ranks = np.concatenate(ranks)
    return ranks


def compute_hierarchical_similarity(df_hierarchical_similarity, id2idx, mat_embedding):
    count = 0
    total_rows = len(df_hierarchical_similarity)
    # Create a dictionary to store the similarity scores
    mat_embedding = torch.tensor(mat_embedding)

    accuracy_hierarchical_similarity = []

    accuracy = 0
    # Iterate through the DataFrame and compute the similarity scores
    for row in df_hierarchical_similarity.iter_rows():
        count += 1
        if count % 100 == 0:
            print(f"Processing row {count}/{total_rows}...")
        sctid = row[0]
        close_sctid = row[1]
        far_sctid = row[2]

        # Get the indices of the concepts
        idx_sctid = id2idx.get(sctid)
        idx_close_sctid = id2idx.get(close_sctid)
        idx_far_sctid = id2idx.get(far_sctid)

        if idx_sctid is not None and idx_close_sctid is not None and idx_far_sctid is not None:
            # Extract embeddings and reshape to 2D tensors
            emb_sctid = mat_embedding[idx_sctid].unsqueeze(0)  # Shape: [1, embedding_dim]
            emb_close_sctid = mat_embedding[idx_close_sctid].unsqueeze(0)  # Shape: [1, embedding_dim]
            emb_far_sctid = mat_embedding[idx_far_sctid].unsqueeze(0)  # Shape: [1, embedding_dim]

            # Compute the similarity scores
            score_close = torch.cosine_similarity(emb_sctid, emb_close_sctid, dim=-1)
            score_far = torch.cosine_similarity(emb_sctid, emb_far_sctid, dim=-1)

            # Append the comparison result
            accuracy_hierarchical_similarity.append(score_close.item() > score_far.item())

    accuracy = sum(accuracy_hierarchical_similarity) / len(accuracy_hierarchical_similarity)
    return accuracy


def compute_semantic_composition(df_semantic_composition, id2idx, embeddings_exp_ft, list_idx_all_pre_set):
    top_k_pre = []
    count = 0
    for row in df_semantic_composition.iter_rows():
        count += 1
        if count % 100 == 0:
            print(f"Processing {count}/{len(df_semantic_composition)}")

        anchor_id = str(row[0])
        if anchor_id not in id2idx:
            continue

        anchor_idx = id2idx[anchor_id]

        try:
            related_ids = ast.literal_eval(row[1])
        except (ValueError, SyntaxError):
            continue

        related_embedding = torch.zeros(embeddings_exp_ft.shape[1], device=embeddings_exp_ft.device)
        count_valid = 0

        for related_id in related_ids:
            related_id = str(related_id)
            if related_id not in id2idx:
                continue
            related_idx = id2idx[related_id]
            related_embedding += embeddings_exp_ft[related_idx]
            count_valid += 1

        if count_valid == 0:
            continue

        related_embedding /= count_valid

        # Compute similarity
        similarity_scores = torch.matmul(related_embedding, embeddings_exp_ft.T)
        sorted_indices = torch.argsort(similarity_scores, descending=True)

        # Filter sorted indices for fully-defined and pre-defined
        sorted_indices_pre = [idx.item() for idx in sorted_indices if idx.item() in list_idx_all_pre_set]

        try:
            rank_pre = sorted_indices_pre.index(anchor_idx)
        except ValueError:
            rank_pre = -1

        top_k_pre.append(rank_pre)

    return np.array(top_k_pre)