import numpy as np
from tqdm import tqdm

def process_and_save_batches(df, embedding_dict, batch_size=100000, prefix="train"):
    X_batch = []
    y_batch = []
    total_saved = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Batching {prefix}")):
        key = (row["protein_id"], row["residue_index"])
        if key in embedding_dict:
            X_batch.append(embedding_dict[key])
            y_batch.append(row["secondary_structure"])

        # Save every batch_size examples
        if len(X_batch) == batch_size:
            save_batch(X_batch, y_batch, total_saved, prefix)
            total_saved += batch_size
            X_batch, y_batch = [], []

    # Save final batch
    if X_batch:
        save_batch(X_batch, y_batch, total_saved, prefix)

def save_batch(X, y, index, prefix):
    np.save(f"{prefix}_X_{index}.npy", np.array(X))
    np.save(f"{prefix}_y_{index}.npy", np.array(y))

import glob

def load_batches(prefix="train"):
    X_all, y_all = [], []

    for x_file in sorted(glob.glob(f"{prefix}_X_*.npy")):
        X_all.append(np.load(x_file))
    for y_file in sorted(glob.glob(f"{prefix}_y_*.npy")):
        y_all.append(np.load(y_file))

    return np.concatenate(X_all), np.concatenate(y_all)
