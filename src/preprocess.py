import numpy as np

from PIL import Image
image_path = '/home/natty/dictionary_learning/data/images/building.jpg'
try:
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)  # Convert to numpy array
    print(f"Image successfully loaded from {image_path}, shape: {image.shape}")
except Exception as e:
    raise ValueError(f"Error loading image with Pillow: {e}")
