import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from preprocess import preprocess_image
from model import MLP 

# Config dyal l-page
st.set_page_config(page_title="MNIST Classifier - Master IGOV", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; background-color: #0047AB; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("Système de Reconnaissance de Chiffres Manuscrits")
st.write("Analyse Matricielle par Perceptron Multi-Couches")
st.markdown("---")

@st.cache_resource
def load_trained_model():
    try:
        # Chargement dyal l-weights
        data = np.load('model_weights.npz')
        
        # Détection automatique dyal architecture
        h_size, i_size = data['W1'].shape
        o_size = data['W2'].shape[0]
        
        # Initialisation
        nn = MLP(input_size=i_size, hidden_size=h_size, output_size=o_size)
        nn.W1, nn.b1 = data['W1'], data['b1']
        nn.W2, nn.b2 = data['W2'], data['b2']
        return nn
    except Exception as e:
        st.error(f"Erreur technique de chargement : {e}")
        return None

model = load_trained_model()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Saisie Manuscrite")
    canvas_result = st_canvas(
        stroke_width=20, stroke_color="#FFFFFF", background_color="#000000",
        height=300, width=300, drawing_mode="freedraw", key="canvas",
    )
    if st.button("Réinitialiser"):
        st.rerun()

if canvas_result.image_data is not None and model is not None:
    raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    processed_data, debug_img = preprocess_image(raw_img)
    
    with col2:
        st.subheader("Résultat de l'IA")
        if st.button('Lancer la classification'):
            # Calcul via model.py
            output = model.forward_propagation(processed_data)
            probs = output.flatten()
            prediction = np.argmax(probs)
            
            # Affichage
            c1, c2 = st.columns(2)
            c1.metric("Chiffre Prédit", prediction)
            c2.metric("Confiance", f"{np.max(probs)*100:.2f}%")
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(10), probs, color=['#0047AB' if i==prediction else '#E0E0E0' for i in range(10)])
            ax.set_xticks(range(10))
            st.pyplot(fig)
            
            with st.expander("Diagnostic Technique"):
                st.write(f"Dimension détectée : {model.input_size} neurones")
                st.image(debug_img, width=150)