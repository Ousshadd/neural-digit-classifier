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
    .stButton>button { widimport numpy as np
from PIL import Image, ImageOps

def preprocess_image(image_data):
    # Convertir en niveaux de gris
    img = image_data.convert('L')
    
    # Inverser si nécessaire (on veut blanc sur noir)
    # Si le dessin est noir sur blanc, on inverse
    if np.mean(img) > 127:
        img = ImageOps.invert(img)
    
    # Trouver la zone contenant le chiffre (Bounding Box)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Redimensionner à 20x20 en gardant le ratio (standard MNIST)
    w, h = img.size
    ratio = 20.0 / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Créer une image noire 28x28 et coller le 20x20 au centre
    final_img = Image.new('L', (28, 28), 0)
    upper = (28 - new_size[1]) // 2
    left = (28 - new_size[0]) // 2
    final_img.paste(img, (left, upper))
    
    # Normalisation et mise à plat pour le MLP
    data = np.array(final_img).reshape(1, 784) / 255.0
    return data, final_imgth: 100%; border-radius: 4px; height: 3em; background-color: #0047AB; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("Système de Reconnaissance de Chiffres Manuscrits")
st.write("Analyse Matricielle par Perceptron Multi-Couches")
st.markdown("---")

@st.cache_resource
def load_trained_model():
    try:

        data = np.load('model_weights.npz')
        
        
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
            
            x_input = processed_data.flatten()
            

            x_input = x_input[:model.input_size]
            
            # 3. Calcul via model.py
            output = model.forward_propagation(x_input)
            # ------------------------
            
            probs = output.flatten()
            prediction = np.argmax(probs)
            
            # Affichage Metrics
            c1, c2 = st.columns(2)
            c1.metric("Chiffre Prédit", prediction)
            c2.metric("Confiance", f"{np.max(probs)*100:.2f}%")
            
            # Histogramme
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(10), probs, color=['#0047AB' if i==prediction else '#E0E0E0' for i in range(10)])
            ax.set_xticks(range(10))
            ax.set_ylabel("Probabilité")
            st.pyplot(fig)
            
            with st.expander("Diagnostic Technique"):
                st.write(f"Dimension du modèle : {model.input_size} neurones")
                st.write(f"Dimension de l'entrée : {x_input.shape[0]} pixels")
                st.image(debug_img, width=150)