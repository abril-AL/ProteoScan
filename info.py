# info.py
import streamlit as st # type: ignore
import os
import base64

def get_base64_image(image_path):
    """Helper to load local images securely in Streamlit markdown if needed."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render_technical_overview():
    """Returns the Markdown content for the Technical Overview page."""
    
    st.title("Architecture & Engineering Choices")
    
    # st.image("assets/architecture_diagram.png", caption="ProteoScan System Architecture")

    st.markdown("""
    ### 1. The Core Pipeline
    ProteoScan is built on a decoupled machine learning architecture, separating heavy model training from lightweight, on-demand inference. 
    * **Feature Extraction:** It utilizes Hugging Face's `facebook/esm2_t6_8M_UR50D` transformer model to generate dense mathematical representations (embeddings) of amino acid sequences.
    * **Classification:** A Scikit-Learn `SGDClassifier` predicts the final secondary structure using these embeddings.
    
    ### 2. Key Engineering Decisions
    
    **Addressing Class Imbalance (Q8 to Q3 Mapping)**
    The original raw DSSP training data contained 8 distinct structural states (e.g., Alpha Helices, 3-10 Helices, Pi Helices, Beta-Bridges). Due to extreme class imbalance in nature (where Alpha Helices are highly common but Pi Helices are rare), a linear model struggles to achieve high accuracy across all 8 classes. 
    * **Solution:** Engineered a mapping pipeline to simplify the dataset into a Q3 predictor: **Helix (H), Sheet (E), and Coil (C)**. This significantly boosts model confidence, mitigates class imbalance, and provides a much cleaner UX.
    
    **Memory Optimization (Dimensionality Slicing)**
    By default, the ESM-2 8M model outputs a 320-dimensional vector for every single amino acid. 
    * **Solution:** During both training and inference, the embeddings are explicitly sliced down to the first 50 dimensions (`[:, :50]`). This drastically reduces the memory footprint and speeds up the batched `SGDClassifier` training loop without sacrificing the foundational contextual data the transformer provides.
    
    **Decoupled Architecture & Serialization**
    To ensure the UI remains fast, the system does not train on the fly. 
    * **Solution:** `train.py` uses `joblib` (optimized for large NumPy arrays) to serialize the trained model weights and LabelEncoder. Streamlit then uses the `@st.cache_resource` decorator in `app.py` to load these artifacts into memory only once upon server startup, reducing inference time from minutes to seconds.
    
    ### 3. Motivation
    
    **Why ProteoScan?**
                
    Finding the structure of a protein took John Kendrew 12 years and earned him a nobel prize.
    The old way, protein crystallization is expensive and time-consuming.
    Today AI can speed up this process from years to seconds. ProteoScan is a proof-of-concept that demonstrates how we can leverage 
    state-of-the-art transformer models and classical machine learning techniques to democratize access to protein structure prediction.
    """)

    col1, col2 = st.columns(2)    
    with col1:
        st.image("assets/turd.png", caption="\"Turd of the Century\"",  width=400)
    with col2:
        st.image("assets/myo.png", caption="Myogloobin Structure", width=300)
    
    st.markdown("""                            
    ### 4. What's it Actually Doing ?
            
    **Primary Structure**
                
    This is the sequence of amino acids in the protein, represented by single-letter codes (e.g., M for Methionine, A for Alanine). 
    It is the most basic level of protein structure and determines all higher levels.
    """ )          
    st.image("assets/ps.png", caption="Primary Structure", width=400)
    st.markdown("""
    **Secondary Structure (What ProteoScan Predicts)**
                
    This refers to local spatial arrangements of the amino acid chain, primarily:            
    - **Alpha Helices (H)**: Coiled structures stabilized by hydrogen bonds.
    - **Beta Sheets (E)**: Flattened, pleated structures formed by hydrogen bonds between different segments of the chain.
    - **Random Coils (C)**: Irregular, flexible regions that do not fit into the other two categories.
    The secondary structure is crucial for understanding the protein's overall shape and function.""")

    st.image("assets/ss.png", caption="Secondary Structure", width=500)
    st.markdown("""
                
    **Tertiary Structure**
                
    This is the three-dimensional folding of the entire protein, determined by interactions between the secondary structure
    elements. It is the most critical level for understanding the protein's function, as it dictates how the protein interacts with other molecules.   
    - This is where AlphaFold, created at DeepMind, comes in. It uses deep learning to predict the tertiary structure of proteins with remarkable accuracy, 
    often rivaling experimental methods. However, AlphaFold is computationally intensive and not easily accessible for quick predictions on arbitrary sequences. 
    ProteoScan fills this gap by providing a fast, lightweight tool for secondary structure prediction, which can be a valuable first step in understanding protein
    function before investing in more resource-heavy tertiary structure predictions. 
    - Now we know almost the protein structures for 200 million proteins, almost all known to exist in nature
    - Helps with developing vaccines, antibiotics, how protein mutations lead to cancer, shizophrenia, and more, all types of life mechanisms
    - Even the possibility for creating new proteins from scratch, which could be used for synthetic antivenoms that are human compatible, or enzymes that can break down plastic.
    
    Learning about alphafold and the way AI can revolutionize science was the original inspiration for this project, I wanted to showcase the
    power of these models in a way that was accessible and interactive for people who might not have a background in computational biology.
    """)
