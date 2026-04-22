# info.py
import streamlit as st # type: ignore
import os
import base64
from info import get_base64_image

def render_presentation():
    """Returns the Markdown content for the Presentation page."""
    
    st.title("ProteoScan: Protein Structure Prediction with Transformers")
    
    st.markdown("""

    **Overview:**
    Let's explore ProteoScan, a machine learning pipeline designed to predict protein secondary structures from amino acid sequences. 
    We will cover the motivation behind the project, the technical architecture, and a live demonstration of the system in action.
    
    ### Agenda:
    1. Introduction to Protein Structure Prediction
    2. The ProteoScan Pipeline
    3. Technical Architecture & Engineering Choices
    4. Live Demonstration
    5. Q&A Session
    
    ### 1. Introduction to Protein Structure Prediction
    
    Proteins are fundamental to all biological processes, and their function is largely determined by their structure. Traditional methods for determining protein structure, such as X-ray crystallography and NMR spectroscopy, are time-consuming and expensive. With advances in machine learning, we can now predict protein structures computationally, significantly accelerating research in fields like drug discovery and molecular biology.
    """)
    st.image("assets/structure.png", width=600)

    st.markdown("""        
    ### 2. The ProteoScan Pipeline
    
    ProteoScan consists of two main phases: training and inference. During training, we use a dataset of amino acid sequences and their corresponding secondary structures to train a machine learning model. For inference, users can input a new amino acid sequence, and the system will predict its secondary structure in real-time.
    
    ### 3. Technical Architecture & Engineering Choices
    
    The core of ProteoScan's architecture is built on Hugging Face's transformer models for feature extraction and Scikit-Learn's `SGDClassifier` for classification. I made several engineering decisions to optimize performance and accuracy, including mapping the original 8-state labels to a more balanced 3-state classification (Q3) and slicing the transformer embeddings to reduce memory usage.
    
    ### 4. Live Demonstration
    
    Now let's see ProteoScan in action! We will input an amino acid sequence and observe how the system predicts its secondary structure in real-time.

    Thank you!
    """)