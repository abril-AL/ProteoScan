# ProteoScan: Secondary Structure Predictor

ProteoScan is an interactive bioinformatics dashboard that takes raw protein sequences, processes them through an ESM-2 and SGDClassifier machine learning pipeline, and utilizes a secondary logic layer to summarize biological motifs alongside a dynamic visual topology map.

Developed from a (more boring) class assignment, this project has been refactored into a decoupled, production-ready software architecture that separates heavy model training from lightweight, on-demand inference.

# Architecture & Directory Overview

* **`app.py`**: The frontend Graphical User Interface (GUI). Built with Streamlit, it handles user input, dynamically loads the pre-trained ML models, and renders custom HTML/SVG topology maps and confidence charts.
* **`train.py`**: The backend model builder. 
    * Loads `train.tsv` and `sequences.fasta` from the `base/` directory.
    * Generates high-dimensional token embeddings using Hugging Face's ESM-2 model.
    * Executes a batched training loop for the `SGDClassifier`.
    * Serializes and saves the final model artifacts to disk using `joblib`.
* **`inference.py`**: The Logic Bridge.
    * Abstracted module that strictly handles ESM-2 tokenization and Scikit-Learn prediction logic, decoupling the machine learning backend from the frontend UI.
    * Features `load_ml_artifacts()` to load the `.joblib` model weights and LabelEncoder into memory.
    * Features `predict_secondary_structure()` which takes a raw string input, processes the 50-dimensional ESM-2 embeddings, and returns Q3 (Helix, Sheet, Coil) predictions alongside confidence probability scores.
* **`base/`**: The data and legacy code directory.
    * Contains the original raw datasets (`train.tsv`, `test.tsv`, `sequences.fasta`).
    * `batching.py`: Helper module for feature extraction and memory management.
    * `legacy_code.py`: The original "all-in-one" academic script that ProteoScan was built from, preserved for reference.
* **`models/`**: *(Generated after running `train.py`)* Stores the saved `sgd_classifier.joblib` and `label_encoder.joblib` files for fast inference loading.

---

## Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn torch transformers biopython matplotlib joblib tqdm
```

### 2. Model Training

Before launching the app, you must train the model and generate the serialization files. This process leverages the ESM-2 transformer and will take some time depending on your hardware.

```bash
python train.py
```

*Note: A developer's slice (e.g., first 500 sequences) can be configured in train.py for rapid local testing. Its just commented out in `train.py`*

### 3. Launching the GUI

Once the models/ directory is populated with your .joblib files, start the Streamlit application:

```Bash
streamlit run app.py
```

### Extra

Heres more on what it looked like to train the model

```bash
=== Phase 1: Data Ingestion ===
Mapping 8-state labels to 3-state (Q3)...
Train samples loaded: 2708235
Total sequences loaded: 8693

=== Phase 2: Feature Extraction (ESM-2) ===
Using device: cpu
Embedding sequences: 100%|███████████████████████████████████████████████████████| 8693/8693 [7:38:43<00:00,  3.17s/seq]

Saved generated embeddings to disk.

=== Phase 3: Batch Generation ===
Batching train: 100%|██████████████████████████████████████████████████████| 2708235/2708235 [04:27<00:00, 10109.70it/s]

=== Phase 4: Model Training ===
Training in batches: 100%|████████████████████████████████████████████████████████████| 107/107 [00:21<00:00,  4.88it/s]

=== Validation Performance ===
              precision    recall  f1-score   support

           .       0.30      0.28      0.29      4380
           B       0.00      0.00      0.00       368
           E       0.57      0.51      0.54      5610
           G       0.60      0.00      0.01      1062
           H       0.50      0.88      0.64      8101
           I       0.00      0.00      0.00       102
           P       0.00      0.00      0.00       576
           S       0.13      0.00      0.01      2008
           T       0.37      0.20      0.26      2793

    accuracy                           0.47     25000
   macro avg       0.27      0.21      0.19     25000
weighted avg       0.42      0.47      0.41     25000


=== Phase 5: Saving Artifacts ===
Success! Model and LabelEncoder saved to the 'models/' directory.
```

Of course i only ran this once because it took over 8 hours! But it was rigouously tested against a smaller set. 

But guess what ! I messed up and used a Q8 model, i messed up the mapping to Q3..
However i luckily didnt have to wait 8 hours again. It's hard to explain but i tweaked 1 line and i just ran it again with the cached embedding. Heres an ai summary bc i cant explain it that clearly tbh..

```
You were able to skip that massive 7.5-hour step because of a fundamental concept in machine learning: the separation of features (your input data) and labels (your target answers).

Here is exactly why the esm_embeddings.pkl file works seamlessly for both Q8 and Q3:
- ESM-2 is "Label Blind": The ESM-2 transformer model's only job is to look at the raw amino acid sequence string (e.g., "MKTLL...") and translate each amino acid into a mathematical vector. It does not look at your .tsv file and does not know anything about secondary structure labels like Helix, Sheet, or Coil.
- The .pkl is Just the Math: Your esm_embeddings.pkl file is simply a saved dictionary of those mathematical sequence vectors. Because the underlying raw sequences did not change, those vectors remain perfectly valid and your previous hours of computation were not wasted.
- Only the Classifier Cares About Q3 vs. Q8: The difference between Q8 (9 states) and Q3 (3 states) only matters during the final training phase when your SGDClassifier tries to learn the mapping between the ESM-2 embeddings and the target labels.

By adding the mapping function to your train.py script, you only changed the target answers (simplifying them to just 'H', 'E', and 'C').

Because generating the embeddings is the heaviest part of the script , caching them to disk allowed your script to instantly load the math , quickly generate fresh .npy batches paired with your new Q3 target labels , and retrain the final classifier in a matter of seconds.
```

But heres what i got after that terrifying experience:

```
=== Phase 1: Data Ingestion ===
Mapping 8-state labels to 3-state (Q3)...
Train samples loaded: 2708235
Total sequences loaded: 8693

=== Phase 2: Feature Extraction (ESM-2) ===
Using device: cpu
Loaded cached embeddings from disk.

=== Phase 3: Batch Generation ===
Batching train: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2708235/2708235 [04:14<00:00, 10631.13it/s]

=== Phase 4: Model Training ===
Training in batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 107/107 [00:09<00:00, 11.88it/s]

=== Validation Performance ===
              precision    recall  f1-score   support

           C       0.61      0.60      0.60      9757
           E       0.68      0.37      0.48      5978
           H       0.60      0.78      0.68      9265

    accuracy                           0.61     25000
   macro avg       0.63      0.59      0.59     25000
weighted avg       0.62      0.61      0.60     25000


=== Phase 5: Saving Artifacts ===
Success! Model and LabelEncoder saved to the 'models/' directory.
```

Q3! Thank heavens! And we can also see better stats (which of course is just due to the simplification.. but ill take it!)