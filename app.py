import streamlit as st # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import random
import base64
import os

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
            fig, ax = plt.subplots(figsize=(10, 2.5))
            x_indices = np.arange(1, len(sequence_input) + 1)
            
            colors = []
            for p in predictions:
                if p == 'H': colors.append('#3b82f6')     # Blue
                elif p == 'E': colors.append('#ef4444')   # Red
                else: colors.append('#9ca3af')            # Gray
                
            ax.bar(x_indices, confidences, color=colors, edgecolor='black')
            ax.set_ylim(0, 1.3)
            ax.set_xlim(0, len(sequence_input) + 1)
            ax.set_xlabel("Residue Index")
            ax.set_xticks(np.arange(5, len(sequence_input) + 1, 5))
            ax.set_ylabel("Model Confidence")
            
            # Legend handles 
            helix_patch = mpatches.Patch(color='#3b82f6', label='Helix (H)')
            sheet_patch = mpatches.Patch(color='#ef4444', label='Sheet (E)')
            coil_patch  = mpatches.Patch(color='#9ca3af', label='Coil (C)')
            ax.legend(handles=[helix_patch, sheet_patch, coil_patch], loc='upper right', ncol=3, framealpha=0.9)

            # Render the plot
            st.pyplot(fig)
            
            # --- Part B: The Sequence with Images ---

            # --- Helper Function for SVG Icons ---
            def get_structure_svg(pred):
                """Returns a raw SVG string based on the predicted structure, minified to prevent Streamlit bugs."""
                if pred == 'H':
                    return "<svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='#3b82f6' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><path d='M4 12 C 4 4, 10 4, 10 12 C 10 20, 16 20, 16 12 C 16 4, 22 4, 22 12' /></svg>"
                elif pred == 'E':
                    return "<svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='#ef4444' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><path d='M3 12 h14 M12 5 l7 7 l-7 7' /></svg>"
                else:
                    return "<svg width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='#9ca3af' stroke-width='2.5' stroke-linecap='round' stroke-dasharray='4 4'><path d='M4 12 h16' /></svg>"

            # --- Part B: The Sequence with Embedded SVGs ---
            st.markdown("#### Sequence Topology Overview")
            
            # Start the flex container (minified)
            colored_sequence_html = "<div style='display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 15px; font-family: monospace; font-size: 20px; font-weight: bold;'>"
            
            for aa, pred in zip(sequence_input, predictions):
                # Determine the correct color for the text
                if pred == 'H': color = "#3b82f6"
                elif pred == 'E': color = "#ef4444"
                else: color = "#9ca3af"
    
                # Fetch the pure SVG code
                svg_icon = get_structure_svg(pred)
    
                # Create the column for each residue (Minified into one line to prevent Markdown breaking)
                colored_sequence_html += f"<div style='display:flex; flex-direction:column; align-items:center; width:28px;' title='Prediction: {pred}'><div style='height:24px; margin-bottom:2px;'>{svg_icon}</div><span style='color:{color};'>{aa}</span></div>"
    
            colored_sequence_html += "</div>"

            # Render the HTML
            st.markdown(colored_sequence_html, unsafe_allow_html=True)
            
            # --- Part C: VISUAL LEGEND ---
            st.markdown("#### Legend")
            
            # Fetch the SVGs for the legend
            helix_svg = get_structure_svg('H')
            sheet_svg = get_structure_svg('E')
            coil_svg = get_structure_svg('C')
            
            # Create a minified flexbox layout for the legend items
            legend_html = "<div style='display: flex; gap: 24px; align-items: center; font-family: sans-serif; font-size: 16px; margin-top: 10px; margin-bottom: 20px;'>"
            
            # Helix Item
            legend_html += f"<div style='display: flex; align-items: center; gap: 8px;'><div style='width: 24px; height: 24px;'>{helix_svg}</div><span style='color: #3b82f6; font-weight: bold;'>Alpha Helix (H)</span></div>"
            
            # Sheet Item
            legend_html += f"<div style='display: flex; align-items: center; gap: 8px;'><div style='width: 24px; height: 24px;'>{sheet_svg}</div><span style='color: #ef4444; font-weight: bold;'>Beta Sheet (E)</span></div>"
            
            # Coil Item
            legend_html += f"<div style='display: flex; align-items: center; gap: 8px;'><div style='width: 24px; height: 24px;'>{coil_svg}</div><span style='color: #9ca3af; font-weight: bold;'>Random Coil (C)</span></div>"
            
            legend_html += "</div>"
            
            # Render the legend
            st.markdown(legend_html, unsafe_allow_html=True)


            # --- 5. BIOLOGICAL INSIGHTS ---
            st.subheader("Biological Insights")
            
            insights = analyze_biology(predictions)
            for insight in insights:
                st.info(insight)
                
            # Display raw string for reference
            st.write(f"**Raw Prediction Sequence:** `{''.join(predictions)}`")

        else:
            st.error("Please enter a valid amino acid sequence containing only letters.")

