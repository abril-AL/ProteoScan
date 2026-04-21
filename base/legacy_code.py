#########################################################################################
#   This page includes the code for the class assignment that ProteoScan is based on.   #
#   This code read in train/test data, generates ESM embeddings, trains a classifier,   #
#   and makes predictions. It will updated for the GUI, but is here to serve as a       #
#   reference for the core logic.                                                       #
#########################################################################################

# Project 1 - CS C121
# Secondary Structure Prediction
import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

import batching

DB = True # Debug

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow info/warning/error logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # HuggingFace warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# -- STEP 1: Data Ingestion --
# train.tsv: Contains protein IDs, amino acid, and secondary structure labels.
# test.tsv: Contains protein IDs and amino acids only.
# sequences.fasta: Contains full sequences of each protein.

train_df = pd.read_csv("project_1/train.tsv", sep='\t')
test_df = pd.read_csv("project_1/test.tsv", sep='\t')

seq_dict = {}
for record in SeqIO.parse("project_1/sequences.fasta", "fasta"):
    protein_id = record.id.strip() 
    sequence = str(record.seq)
    seq_dict[protein_id] = sequence

if DB:
    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))
    print("Loaded sequences:", len(seq_dict))
    print("Sample:", list(seq_dict.items())[0])

def parse_id(entry):
    # e.g., "3KVH_LYS_6" → ("3KVH", 6)
    parts = entry.split('_')
    return parts[0], int(parts[2])

for df in [train_df, test_df]:
    id_parts = df['id'].str.split('_', expand=True)
    df['protein_id'] = id_parts[0]
    df['residue_index'] = id_parts[2].astype(int)

# --- STEP 2: Feature Extraction ---

# load esm model
if DB:
    print("Loading ESM model")
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# try to use GPU
if DB:
    print("Checking for GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate ESM embeddings - tokenize protein sequence
embedding_cache_file = "esm_embeddings.pkl"
if os.path.exists(embedding_cache_file):
    # cache so i dont kill my laptop
    with open(embedding_cache_file, "rb") as f:
        embedding_dict = pickle.load(f)
    print("Loaded cached embeddings from file.")
else:
    print("Embedding proteins...")
    embedding_dict = {}
    for protein_id, sequence in tqdm(seq_dict.items(), desc="Embedding proteins", unit="seq"):
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Shape: (seq_len, hidden_dim)
        token_embeddings = outputs.last_hidden_state[0, 1:-1, :]  # slice off start/end tokens

        for i, emb in enumerate(token_embeddings):
            embedding_dict[(protein_id, i + 1)] = emb.cpu().numpy()[:50]  # 1-based indexing

    # Save embeddings to disk
    with open(embedding_cache_file, "wb") as f:
        pickle.dump(embedding_dict, f)
    print("Saved embeddings to disk.")

# --- STEP 3: Split Data Set ---

# extract features and labels from training data
if DB:
    print("Extracting Features...")
batching.process_and_save_batches(train_df, embedding_dict, batch_size=25_000)

# Step 3.5: Set aside one batch for validation, train on the rest
x_batches = sorted(glob.glob("train_X_*.npy"))
y_batches = sorted(glob.glob("train_y_*.npy"))
total_batches = len(x_batches)

# Use last batch for validation
val_X = np.load(x_batches[-1])
val_y = np.load(y_batches[-1])

label_encoder = LabelEncoder()
val_y_encoded = label_encoder.fit_transform(val_y)

# --- STEP 4: Train Classifier --- 
if DB:
    print("Training SGD Classifier...")

# Train on all other batches
clf = SGDClassifier(loss='log_loss',  
                    max_iter=1,       
                    learning_rate='optimal',
                    warm_start=True)

for i in range(total_batches - 1):  # Leave one batch for validation
    X = np.load(x_batches[i])
    y = np.load(y_batches[i])
    y_encoded = label_encoder.transform(y)

    if i == 0:
        clf.partial_fit(X, y_encoded, classes=np.unique(val_y_encoded))  # first batch must define all classes
    else:
        clf.partial_fit(X, y_encoded)

def batched_predict(clf, X_val, batch_size=100000):
    preds = []
    for i in tqdm(range(0, len(X_val), batch_size), desc="Predicting in batches"):
        batch = X_val[i:i + batch_size]
        batch_preds = clf.predict(batch)
        preds.append(batch_preds)
    return np.concatenate(preds)

y_pred = batched_predict(clf, val_X)

print("\n=== Validation Performance ===")
print(classification_report(val_y_encoded, y_pred, target_names=label_encoder.classes_))

# --- STEP 5: Made Predictions ---
predictions = []
missing = 0

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting test.tsv"):
    key = (row["protein_id"], row["residue_index"])

    if key in embedding_dict:
        embedding = embedding_dict[key]
        
        # If using feature scaling during training, apply it here too
        embedding = np.array(embedding).reshape(1, -1)

        pred = clf.predict(embedding)
        pred_label = label_encoder.inverse_transform(pred)[0]
        predictions.append(pred_label)
    else:
        missing += 1
        predictions.append('.')  # placeholder or fallback prediction

print(f"Done predcting. Missing embeddings: {missing}")

# Add predictions to test_df and save
test_df["prediction"] = predictions

# --- STEP 6: Export or Codabench Submission ---

# Output: match Codabench format — id and prediction, no header, tab-separated
test_df[["id", "prediction"]].to_csv("predictions.csv", sep="\t", index=False, header=False)