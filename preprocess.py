import numpy as np
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
    return data, final_img