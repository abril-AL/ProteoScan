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