import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image_data):
    # 1. Convertir en niveaux de gris
    img = image_data.convert('L')
    
    # 2. Inverser si nécessaire (on veut blanc sur noir)
    if np.mean(img) > 127:
        img = ImageOps.invert(img)
    
    # 3. Centrage (Bounding Box)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # 4. Redimensionner à 20x20 (Standard MNIST logic)
    w, h = img.size
    ratio = 20.0 / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # 5. Créer une image 28x28 (Padding noir)
    final_img = Image.new('L', (28, 28), 0)
    upper = (28 - new_size[1]) // 2
    left = (28 - new_size[0]) // 2
    final_img.paste(img, (left, upper))
    
    # 6. Normalisation et Formatage
    # IMPORTANT: On utilise .reshape(784, 1) pour être compatible avec np.dot(W, X)
    data = np.array(final_img).astype(np.float32).reshape(784, 1) / 255.0
    
    return data, final_img