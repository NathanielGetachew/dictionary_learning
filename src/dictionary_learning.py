import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt

# Function for sparse coding
def sparse_coding(patches, dictionary, n_nonzero_coefs=3):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    return omp.fit(dictionary.T, patches).transform(dictionary.T)

# Dictionary learning function
def dictionary_learning(patches, dictionary_init, n_iter=50):
    dictionary_learning_model = DictionaryLearning(n_components=dictionary_init.shape[1], 
                                                  max_iter=n_iter, 
                                                  tol=1e-6)
    
    # Learn the dictionary, it will initialize from random values
    dictionary_learning_model.fit(patches)

    # Override the learned dictionary with the initial dictionary
    dictionary_learning_model.components_ = dictionary_init.T

    return dictionary_learning_model.components_

# Load the preprocessed data
patches_normalized = np.load('/home/natty/dictionary_learning/data/patches_normalized.npy')  # Load your preprocessed data here

# Initialize the dictionary
patch_size = 8 * 8  # Flattened size of each 8x8 patch
n_atoms = 100  # Number of dictionary atoms
dictionary_init = np.random.randn(patch_size, n_atoms)

# Perform dictionary learning
dictionary_final = dictionary_learning(patches_normalized, dictionary_init, n_iter=50)

# Save the learned dictionary as a PNG file
plt.figure(figsize=(10, 10))
for i in range(n_atoms):
    plt.subplot(10, 10, i + 1)
    plt.imshow(dictionary_final[i].reshape(8, 8), cmap='gray')
    plt.axis('off')

# Save the figure as a PNG
plt.tight_layout()
plt.savefig('/home/natty/dictionary_learning/dictionary_learning_output.png', dpi=300)

# Close the plot to free memory
plt.close()
