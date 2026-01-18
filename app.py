import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np
import joblib
from preprocess import preprocess_image
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Classification MNIST", layout="wide")

# Style CSS pour affiner l'esthétique
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; }
    .reportview-container { background: #fafafa; }
    </style>
    """, unsafe_allow_html=True)

st.title("Système de Reconnaissance de Chiffres Manuscrits")
st.markdown("---")

# Charger le modèle (MLP)
@st.cache_resource
def load_my_model():
    return joblib.load('model_mnist.joblib')

model = load_my_model()

# Division de l'interface
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Zone de Saisie")
    st.write("Veuillez dessiner un chiffre ci-dessous :")
    canvas_result = st_canvas(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Réinitialiser"):
        st.rerun()

if canvas_result.image_data is not None:
    raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    
    # Prétraitement
    processed_data, debug_img = preprocess_image(raw_img)
    
    with col2:
        st.subheader("Analyse Prédictive")
        
        if st.button('Exécuter la classification'):
            # Prédiction
            probs = model.predict_proba(processed_data)[0]
            prediction = np.argmax(probs)
            confiance = np.max(probs)
            
            # Affichage des indicateurs clés
            c1, c2 = st.columns(2)
            c1.metric("Chiffre Prédit", prediction)
            c2.metric("Indice de Confiance", f"{confiance*100:.2f}%")
            
            st.write("Distribution des probabilités :")
            st.progress(float(confiance))
            
            # Graphique Matplotlib
            fig, ax = plt.subplots(figsize=(6, 3.5))
            digits = list(range(10))
            # Utilisation d'une palette de couleurs sobre
            colors = ['#E0E0E0' if x != prediction else '#0047AB' for x in digits]
            
            ax.bar(digits, probs, color=colors)
            ax.set_xticks(digits)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Probabilité")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            
            # Section de diagnostic technique
            with st.expander("Diagnostic : Vision du modèle"):
                st.write("Image normalisée (28x28 pixels) :")
                st.image(debug_img, width=120)

# Barre latérale informative (Audit et Gouvernance)
st.sidebar.header("Informations Système")
st.sidebar.markdown("""
**Architecture du modèle** Multi-Layer Perceptron (MLP)

**Dataset d'entraînement** MNIST Original (70,000 images)

**Format de sérialisation** Scikit-learn Joblib / NPZ Weights
""")