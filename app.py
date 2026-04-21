import streamlit as st # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import random
import base64
import os
import inference

# --- 1. SETTING UP MODELS ---
@st.cache_resource
def get_models():
    """Loads the ML artifacts once and keeps them in memory."""
    try:
        return inference.load_ml_artifacts()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# Initialize models
tok, esm, clf, le, dev = get_models()

# --- 2. CONFIG & UI ---
st.set_page_config(page_title="ProteoScan", layout="wide")
st.title("🧬 ProteoScan: Secondary Structure Predictor")

# Sidebar for input
with st.sidebar:
    st.header("Input Sequence")
    sequence_input = st.text_area(
        "Paste Protein Sequence (AA):", 
        "MKTLLILVLCFLPLAALG", # Default test sequence
        height=150
    ).upper().strip().replace("\n", "")
    
    analyze_btn = st.button("Analyze Sequence", type="primary")

# --- 3. MAIN LOGIC ---
if analyze_btn:
    if not tok:
        st.warning("Models not loaded. Please ensure train.py has finished and created the 'models' folder.")
    elif not sequence_input:
        st.warning("Please enter a sequence first.")
    else:
        with st.spinner("Predicting secondary structure using ESM-2..."):
            # CALL THE REAL INFERENCE
            predictions, confidences = inference.predict_secondary_structure(
                sequence_input, tok, esm, clf, le, dev
            )
            
        # Create Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # --- Part A: The Plot ---
            st.subheader("Confidence & Structure Map")
            fig, ax = plt.subplots(figsize=(10, 3))
            x_indices = np.arange(1, len(sequence_input) + 1)
            
            # Map colors for the plot
            plot_colors = []
            for p in predictions:
                if p == 'H': plot_colors.append('#3b82f6') # Blue
                elif p == 'E': plot_colors.append('#ef4444') # Red
                else: plot_colors.append('#9ca3af') # Gray
            
            ax.bar(x_indices, confidences, color=plot_colors, edgecolor='black')
            ax.set_ylim(0, 1.3)
            ax.set_xlabel("Residue Index")
            ax.set_xticks(np.arange(5, len(sequence_input) + 1, 5))
            ax.set_ylabel("Confidence")
            
            # Internal Legend
            h_patch = mpatches.Patch(color='#3b82f6', label='Helix (H)')
            e_patch = mpatches.Patch(color='#ef4444', label='Sheet (E)')
            c_patch = mpatches.Patch(color='#9ca3af', label='Coil (C)')
            ax.legend(handles=[h_patch, e_patch, c_patch], loc='upper right', ncol=3)
            
            st.pyplot(fig)

        # --- Part B: The Visual Topology ---
        st.markdown("### Predicted Topology")
        
        # This reuses your SVG helper and minified HTML logic
        colored_sequence_html = "<div style='display:flex; flex-wrap:wrap; gap:4px; font-family:monospace; font-size:20px; font-weight:bold;'>"
        for aa, pred in zip(sequence_input, predictions):
            color = "#3b82f6" if pred == 'H' else "#ef4444" if pred == 'E' else "#9ca3af"
            svg_icon = inference.get_structure_svg(pred) # We should move the SVG helper to inference.py!
            colored_sequence_html += f"<div style='display:flex; flex-direction:column; align-items:center; width:28px;'><div style='height:24px;'>{svg_icon}</div><span style='color:{color};'>{aa}</span></div>"
        colored_sequence_html += "</div>"
        
        st.markdown(colored_sequence_html, unsafe_allow_html=True)
