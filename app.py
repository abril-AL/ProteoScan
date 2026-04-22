import streamlit as st # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import random
import base64
import os
import inference
from google import genai
from google.genai import errors  
import info
import present

# --- SETTING UP MODELS ---
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

# --- CONFIG & UI ---
st.set_page_config(page_title="ProteoScan", layout="wide")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["ProteoScan App", "Technical Overview", "Presentation"])

if page == "ProteoScan App":
    st.title("🧬 ProteoScan: Secondary Structure Predictor")

    # Sidebar for input
    with st.sidebar:
        st.header("Input Sequence")
        sequence_input = st.text_area(
            "Paste Protein Sequence (AA):", 
            "MKIATYNVRVDTEYDQDWQWSFRKEAVCQLINFHDWSLCCIQEVRPNQVRDLKAYTTFTCLSAEREGDGQGEGLAILYNEQKVQAII", # Default test sequence
            height=150
        ).upper().strip().replace("\n", "")
        
        analyze_btn = st.button("Analyze Sequence", type="primary")

    # --- MAIN LOGIC ---
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
                
            # --- The Visual Topology ---
            st.divider()
            st.markdown("### Predicted Topology")
            
            colored_sequence_html = "<div style='display:flex; flex-wrap:wrap; gap:8px; font-family:monospace; font-size:26px; font-weight:bold;'>"
            for aa, pred in zip(sequence_input, predictions):
                color = "#3b82f6" if pred == 'H' else "#ef4444" if pred == 'E' else "#9ca3af"
                svg_icon = inference.get_structure_svg(pred)
                colored_sequence_html += f"<div style='display:flex; flex-direction:column; align-items:center; width:36px;'><div style='height:32px;'>{svg_icon}</div><span style='color:{color};'>{aa}</span></div>"
            colored_sequence_html += "</div>"
            
            st.markdown(colored_sequence_html, unsafe_allow_html=True)

            # --- The Plot ---
            st.divider()
            st.subheader("Confidence & Structure Map")
            fig, ax = plt.subplots(figsize=(10, 1.5))
            x_indices = np.arange(1, len(sequence_input) + 1)
            
            plot_colors = []
            for p in predictions:
                if p == 'H': plot_colors.append('#3b82f6') 
                elif p == 'E': plot_colors.append('#ef4444') 
                else: plot_colors.append('#9ca3af') 
                
            ax.bar(x_indices, confidences, color=plot_colors, edgecolor='none')
            ax.set_ylim(0, 1.3)
            ax.set_xlabel("Residue Index")
            ax.set_xticks([])
            ax.set_ylabel("Confidence")
                
            h_patch = mpatches.Patch(color='#3b82f6', label='Helix (H)')
            e_patch = mpatches.Patch(color='#ef4444', label='Sheet (E)')
            c_patch = mpatches.Patch(color='#9ca3af', label='Coil (C)')
            ax.legend(handles=[h_patch, e_patch, c_patch], loc='upper right', ncol=3)
                
            st.pyplot(fig)

            # --- AI BIO-SUMMARY ---
            st.divider()
            st.subheader("AI Biological Summary")
                
            api_key = st.secrets.get("GEMINI_API_KEY")
                
            if not api_key:
                st.info("Add a Gemini API key to your secrets.toml to unlock AI insights.")
            else:
                with st.spinner("Analyzing structural motifs..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        prompt = f"""
                        You are an expert computational biologist. I have run an amino acid sequence through an ESM-2 and SGDClassifier pipeline to predict its secondary structure (Q3). 
                        H = Alpha Helix, E = Beta Sheet, C = Random Coil.
                        
                        Raw Sequence: {sequence_input}
                        Predicted Structure: {''.join(predictions)}
                            
                        Provide a concise, 2-3 paragraph biological analysis. 
                        1. Classify the overall structural composition.
                        2. Identify any notable motifs.
                        3. Hypothesize a potential broad biological function based on this structure.
                        
                        Keep the tone highly academic, objective, and formatting clean.
                        """
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        st.markdown(response.text)
                    except errors.APIError as e:
                        if e.code == 429:
                            st.warning("**API Rate Limit Reached:** We have hit our 20 requests/day quota. Please try again tomorrow!")
                        else:
                            st.error(f"**Gemini API Error:** {e}")
                    except Exception as e:
                        st.error(f"**An unexpected error occurred:** {e}")
elif page == "Technical Overview":
    info.render_technical_overview()
elif page == "Presentation":
    present.render_presentation()