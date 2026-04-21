# inference.py
import os
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel

def load_ml_artifacts(model_dir="models"):
    """
    Loads the Hugging Face ESM-2 model and the saved Scikit-Learn artifacts.
    Returns the tokenizer, model, classifier, label encoder, and compute device.
    """
    print("Loading ESM-2 Tokenizer and Model...")
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model.to(device)
    esm_model.eval() # Set to evaluation mode
    
    print("Loading Scikit-Learn Classifier and LabelEncoder...")
    clf_path = os.path.join(model_dir, "sgd_classifier.joblib")
    le_path = os.path.join(model_dir, "label_encoder.joblib")
    
    if not os.path.exists(clf_path) or not os.path.exists(le_path):
        raise FileNotFoundError(f"Missing model files in {model_dir}/. Did train.py finish running?")
        
    clf = joblib.load(clf_path)
    le = joblib.load(le_path)
    
    return tokenizer, esm_model, clf, le, device

def predict_secondary_structure(sequence: str, tokenizer, esm_model, clf, le, device):
    """
    Takes a raw amino acid sequence and returns the structural predictions and confidences.
    """
    # 1. Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 2. Get ESM-2 Embeddings
    with torch.no_grad():
        outputs = esm_model(**inputs)
        
    # Shape: (1, seq_len + 2, hidden_dim) -> Slice off the <cls> and <eos> tokens
    token_embeddings = outputs.last_hidden_state[0, 1:-1, :]
    
    # Move to CPU and truncate to 50 dimensions (Crucial: Must match train.py!)
    embeddings_50d = token_embeddings.cpu().numpy()[:, :50]
    
    # 3. Predict Structure using the SGDClassifier
    # predict_proba returns an array of shape (seq_len, num_classes)
    probabilities = clf.predict_proba(embeddings_50d)
    
    # Get the highest probability for each amino acid
    max_probs = np.max(probabilities, axis=1)
    predicted_indices = np.argmax(probabilities, axis=1)
    
    # Convert numerical indices back to 'H', 'E', 'C'
    predictions = le.inverse_transform(predicted_indices)
    
    # Return as standard Python lists for the Streamlit UI
    return list(predictions), list(max_probs)

# --- Quick Local Test ---
# If you run `python inference.py` directly, it will test the bridge.
if __name__ == "__main__":
    try:
        tok, esm, clf, le, dev = load_ml_artifacts()
        test_seq = "MKTLLILVLCFLPLAALG"
        preds, confs = predict_secondary_structure(test_seq, tok, esm, clf, le, dev)
        
        print(f"\nSequence:   {test_seq}")
        print(f"Prediction: {''.join(preds)}")
        print(f"Confidence: {np.mean(confs):.2%} average")
        
    except FileNotFoundError as e:
        print(f"Hold tight: {e}")