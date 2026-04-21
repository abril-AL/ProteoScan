# train.py
import os
import glob
import torch
import pickle
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from base import batching 

DB = True # Debug mode

# Suppress warnings for clean console output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def main():
    print("=== Phase 1: Data Ingestion ===")
    # Only load the training data. We do not care about test.tsv here.
    train_df = pd.read_csv("base/train.tsv", sep='\t')
    
    print("Mapping 8-state labels to 3-state (Q3)...")
    def map_to_q3(label):
        if label in ['H', 'G', 'I']: 
            return 'H'  # Helices
        elif label in ['E', 'B']: 
            return 'E'  # Beta Sheets/Bridges
        else: 
            return 'C'  # Everything else (Turns, Bends, Random Coils)
            
    # Apply mapping
    train_df['label'] = train_df['secondary_structure'].apply(map_to_q3)

    seq_dict = {}
    for record in SeqIO.parse("base/sequences.fasta", "fasta"):
        protein_id = record.id.strip() 
        sequence = str(record.seq)
        seq_dict[protein_id] = sequence

    if DB:
        print(f"Train samples loaded: {len(train_df)}")
        print(f"Total sequences loaded: {len(seq_dict)}")

    # Parse IDs
    id_parts = train_df['id'].str.split('_', expand=True)
    train_df['protein_id'] = id_parts[0]
    train_df['residue_index'] = id_parts[2].astype(int)

    print("\n=== Phase 2: Feature Extraction (ESM-2) ===")
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    embedding_cache_file = "esm_embeddings.pkl"
    if os.path.exists(embedding_cache_file):
        with open(embedding_cache_file, "rb") as f:
            embedding_dict = pickle.load(f)
        print("Loaded cached embeddings from disk.")
    else:
        print("Embedding proteins...")
        embedding_dict = {}
        # USE THIS ONE FOR REAL MODEL
        for protein_id, sequence in tqdm(seq_dict.items(), desc="Embedding sequences", unit="seq"):
        # SLICE: Only take the first 500 items to speed up development bc i dont have 8 hours
        #for protein_id, sequence in tqdm(list(seq_dict.items())[:100], desc="Embedding sequences", unit="seq"):
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Shape: (seq_len, hidden_dim) -> slice off start/end tokens
            token_embeddings = outputs.last_hidden_state[0, 1:-1, :]

            for i, emb in enumerate(token_embeddings):
                embedding_dict[(protein_id, i + 1)] = emb.cpu().numpy()[:50]  # 1-based indexing

        with open(embedding_cache_file, "wb") as f:
            pickle.dump(embedding_dict, f)
        print("Saved generated embeddings to disk.")

    print("\n=== Phase 3: Batch Generation ===")
    batching.process_and_save_batches(train_df, embedding_dict, batch_size=25_000)

    x_batches = sorted(glob.glob("train_X_*.npy"))
    y_batches = sorted(glob.glob("train_y_*.npy"))
    total_batches = len(x_batches)

    # Set aside validation set (last batch)
    val_X = np.load(x_batches[-1])
    val_y = np.load(y_batches[-1])

    label_encoder = LabelEncoder()
    val_y_encoded = label_encoder.fit_transform(val_y)

    print("\n=== Phase 4: Model Training ===")
    clf = SGDClassifier(loss='log_loss',  
                        max_iter=1,       
                        learning_rate='optimal',
                        warm_start=True)

    for i in tqdm(range(total_batches - 1), desc="Training in batches"):
        X = np.load(x_batches[i])
        y = np.load(y_batches[i])
        y_encoded = label_encoder.transform(y)

        if i == 0:
            clf.partial_fit(X, y_encoded, classes=np.unique(val_y_encoded))
        else:
            clf.partial_fit(X, y_encoded)

    # Validate
    print("\n=== Validation Performance ===")
    preds = []
    for i in range(0, len(val_X), 100000): # Batched predict for validation memory management
        batch = val_X[i:i + 100000]
        preds.append(clf.predict(batch))
    y_pred = np.concatenate(preds)
    
    print(classification_report(val_y_encoded, y_pred, target_names=label_encoder.classes_))

    print("\n=== Phase 5: Saving Artifacts ===")
    # Create a directory for our models so the folder stays organized
    os.makedirs("models", exist_ok=True)
    
    # Save the classifier and the label encoder
    joblib.dump(clf, "models/sgd_classifier.joblib")
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    
    print("Success! Model and LabelEncoder saved to the 'models/' directory.")

if __name__ == "__main__":
    main()