# ProteoScan
GUI that takes raw protein sequences, runs it through an ESM-2/SGDClassifier pipeline, and then uses a secondary logic layer to summarize the biology and generate a visual representation.

# Overview

`base` directory
- includes files related to the original class assignment that ProteoScan builds off of
- has the data that was used to train and test the model
- `legacy_code.py` the "all-in-one" script that trains and predicts
    - calls `batching.py` for feature extraction

`app.p`
- runs the ProteoScan streamlit app / GUI

`train.py`
- Responsibilities:
    - Load train.tsv and sequences.fasta.
    - Generate or load the ESM-2 embeddings for the training set.
    - Run the batched training loop for the SGDClassifier.
- at the end, uses `joblib` and `pickle` to save the trained model
