import os
import numpy as np
from PIL import Image

def load_images(image_dir, image_size=(150, 150)):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            img = Image.open(image_path)
            img = img.resize(image_size)  # Resize image to a consistent size
            images.append(np.array(img))
    return np.array(images)

def extract_patches(images, patch_size=(8, 8), stride=4):
    patches = []
    for image in images:
        h, w, _ = image.shape
        for i in range(0, h - patch_size[0] + 1, stride):
            for j in range(0, w - patch_size[1] + 1, stride):
                patch = image[i:i + patch_size[0], j:j + patch_size[1], :]
                patches.append(patch.flatten())  # Flatten the patch into a vector
    return np.array(patches)

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

if __name__ == '__main__':
    image_dir = '/home/natty/dictionary_learning/data'  # Replace with your actual image directory path
    images = load_images(image_dir)
    patches = extract_patches(images)
    patches_normalized = normalize_data(patches)
    
    # Save the normalized patches to a file
    print(patches_normalized.shape)
    np.save('/home/natty/dictionary_learning/data/patches_normalized.npy', patches_normalized)
    print("Data preprocessing completed and saved as patches_normalized.npy.")
