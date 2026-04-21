import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# --- 1. MOCK INFERENCE FUNCTION ---
# TODO: replace with pred/conf logic
def mock_predict_structure(sequence):
    structures = ['H', 'E', 'C']
    predictions = []
    confidences = []
    
    for _ in sequence:
        # arbitrary weights for now: 50% Helix, 25% Sheet, 25% Coil
        pred = random.choices(structures, weights=[0.5, 0.25, 0.25])[0] 
        predictions.append(pred)
        # Random confidence between 60% and 99%
        confidences.append(random.uniform(0.6, 0.99)) 
    return predictions, confidences

# --- 2. BIOLOGICAL RULE ENGINE ---
# TODO: add AI driven insights / summary
def analyze_biology(predictions):
    pred_string = "".join(predictions)
    insights = []
    
    # 10+ Helices in a row
    if "HHHHHHHHHH" in pred_string:
        insights.append("**Potential Transmembrane Domain Detected:** A continuous Alpha-Helix sequence of 10+ residues was found. This strongly indicates membrane-spanning stability.")
    
    if "EEEE" in pred_string:
        insights.append("**Beta-Sheet Motif:** Clustered Beta-strands detected, potentially indicating a structural core or amyloid-forming region.")
        
    if not insights:
        insights.append("No major distinct motifs detected in this specific sequence run.")
        
    return insights

# --- 3. STREAMLIT UI SETUP ---
st.set_page_config(page_title="ProteoScan", page_icon="🧬", layout="wide")

st.title("ProteoScan: Secondary Structure Predictor")
st.markdown("Analyze raw protein sequences using an ESM-2 and SGDClassifier pipeline.")

# Create a two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Sequence Input")
    # Default sequence is 20 residues to test the visuals easily
    sequence_input = st.text_area(
        "Enter Amino Acid Sequence:", 
        value="MKTLLILVLCFLPLAALGGG", 
        height=150
    ).strip().upper()
    
    analyze_button = st.button("Analyze Structure", type="primary")

with col2:
    st.header("Analysis Results")
    
    if analyze_button:
        if sequence_input and sequence_input.isalpha():
            
            # Simulated Processing State
            with st.spinner("Running ESM-2 Embedding & Inference..."):
                time.sleep(1.5) # Simulating processing time
                
                # Get mock predictions
                predictions, confidences = mock_predict_structure(sequence_input)
                
            st.success("Inference Complete!")
            
            # --- 4. VISUALIZATION ---
            st.subheader("Secondary Structure Map")
            
            # --- Part A: Prediction/Confidence ---
            fig, ax = plt.subplots(figsize=(10, 2))
            x_indices = np.arange(1, len(sequence_input) + 1)
            
            colors = []
            for p in predictions:
                if p == 'H': colors.append('#3b82f6')     # Blue
                elif p == 'E': colors.append('#ef4444')   # Red
                else: colors.append('#9ca3af')            # Gray
                
            ax.bar(x_indices, confidences, color=colors, edgecolor='black')
            ax.set_ylim(0, 1.1)
            ax.set_xlim(0, len(sequence_input) + 1)
            ax.set_xlabel("Residue Index")
            ax.set_ylabel("Model Confidence")
            
            # Render the plot
            st.pyplot(fig)
            
            # --- Part B: The Colored Sequence Text ---
            st.markdown("#### Sequence Topology Overview")
            
            # We build this string without multi-line indentation to prevent the code-block bug
            colored_sequence_html = "<div style='font-family: monospace; font-size: 24px; letter-spacing: 6px; margin-bottom: 15px;'>"
            
            for aa, pred in zip(sequence_input, predictions):
                if pred == 'H': 
                    color = "#3b82f6" # Blue
                elif pred == 'E': 
                    color = "#ef4444" # Red
                else: 
                    color = "#9ca3af" # Gray
                
                # Add a 'title' so if you hover over the letter it shows the prediction
                colored_sequence_html += f"<span style='color: {color}; font-weight: bold;' title='Prediction: {pred}'>{aa}</span>"
                
            colored_sequence_html += "</div>"
            
            # Render the colored text
            st.markdown(colored_sequence_html, unsafe_allow_html=True)
            
            # Simple Legend
            st.markdown("""
            **Legend:** &nbsp;&nbsp;
            <span style='color:#3b82f6;'><b>Blue (H)</b></span> = Alpha Helix &nbsp;&nbsp;|&nbsp;&nbsp; 
            <span style='color:#ef4444;'><b>Red (E)</b></span> = Beta Sheet &nbsp;&nbsp;|&nbsp;&nbsp; 
            <span style='color:#9ca3af;'><b>Gray (C)</b></span> = Random Coil
            """, unsafe_allow_html=True)
            
            # --- 5. BIOLOGICAL INSIGHTS ---
            st.subheader("Biological Insights")
            
            insights = analyze_biology(predictions)
            for insight in insights:
                st.info(insight)
                
            # Display raw string for reference
            st.write(f"**Raw Prediction Sequence:** `{''.join(predictions)}`")

        else:
            st.error("Please enter a valid amino acid sequence containing only letters.")