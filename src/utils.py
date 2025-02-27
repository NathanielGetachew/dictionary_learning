import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import cv2

# Function to load an image (you can replace this with your preferred method)
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to extract patches from an image
def extract_image_patches(image, patch_size=(8, 8), max_patches=1000, save_path=None):
    # Extract patches
    patches = extract_patches_2d(image, patch_size, max_patches=max_patches)
    patches = patches.reshape(patches.shape[0], -1)  # Flatten each patch into a 1D array
    
    # Save patches to a .npy file if save_path is provided
    if save_path:
        np.save(save_path, patches)
    
    return patches

# Function to normalize the patches
def normalize_patches(patches):
    return patches / np.linalg.norm(patches, axis=1, keepdims=True)  # L2 normalization
