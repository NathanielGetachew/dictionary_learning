import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from PIL import Image  # Importing Pillow for image loading
from utils import extract_image_patches, normalize_patches  # Importing from utils.py

# Define the patch size before usage
patch_size = (16, 16)  # Define the patch size as a tuple
n_features = patch_size[0] * patch_size[1]  # Number of features per patch (flattened)

# Function for sparse coding using Orthogonal Matching Pursuit (OMP)
def sparse_coding(patches, dictionary, n_nonzero_coefs):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    sparse_codes = omp.fit(dictionary.T, patches.T).coef_.T  # Get the sparse codes (coefficients)
    return sparse_codes

# Function for dictionary learning
def dictionary_learning(patches, dictionary_init, n_iter=10):
    dictionary_learning_model = DictionaryLearning(n_components=dictionary_init.shape[1], max_iter=n_iter, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=10)
    dictionary_final = dictionary_learning_model.fit(patches).components_
    return dictionary_final

# Function to reconstruct images from patches and sparse codes
def reconstruct_image(sparse_codes, dictionary, image_shape):
    reconstructed_patches = np.dot(sparse_codes, dictionary)
    reconstructed_image = np.zeros(image_shape)
    patch_dim = int(np.sqrt(dictionary.shape[1]))  # Assuming square patches
    idx = 0
    for i in range(0, image_shape[0], patch_dim):
        for j in range(0, image_shape[1], patch_dim):
            reconstructed_image[i:i+patch_dim, j:j+patch_dim] = reconstructed_patches[idx].reshape(patch_dim, patch_dim)
            idx += 1
    return reconstructed_image

# Load patches from file or extract them
patches_normalized = None
patches_file_path = '/home/natty/dictionary_learning/data/patches/patches.npy'

# Check if the patches are already saved
image = None  # Initialize image variable
try:
    patches_normalized = np.load(patches_file_path)
except FileNotFoundError:
    # If patches are not found, extract them from the image
    image_path = '/home/natty/dictionary_learning/data/images/28.jpg'
    
    # Load image using Pillow (instead of OpenCV)
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
    except Exception as e:
        print(f"Error loading image: {e}")
    
    if image is None:
        print("Error: Image not loaded.")
    else:
        print(f"Image loaded successfully from {image_path}, shape: {image.size}")
    
    # Convert to numpy array
    image = np.array(image)
    
    # Extract and save patches using the function from utils.py
    patches = extract_image_patches(image, patch_size=patch_size, max_patches=1000, save_path=patches_file_path)
    
    if patches is None or len(patches) == 0:
        raise ValueError("Error extracting patches. Ensure the image and patch size are appropriate.")
    
    # Normalize patches using the function from utils.py
    patches_normalized = normalize_patches(patches)

# Ensure image is loaded before using it
if image is None:
    raise ValueError("Image not loaded. Ensure the image path is correct or patches are loaded.")

# Initialize dictionary (for example, random initialization)
n_components = 100  # Define number of dictionary atoms
dictionary_init = np.random.rand(n_components, n_features)  # Initialize the dictionary with correct dimensions

# Perform dictionary learning
dictionary_final = dictionary_learning(patches_normalized, dictionary_init, n_iter=50)

# Sparse coding
sparse_codes = sparse_coding(patches_normalized, dictionary_final, n_nonzero_coefs=10)

# Reconstruct image from sparse codes and learned dictionary
reconstructed_image = reconstruct_image(sparse_codes, dictionary_final, image.shape)

# Display original and reconstructed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')

# Save the result as a PNG file
plt.savefig('./output/reconstructed_image.png')
plt.close()
