import numpy as np
import matplotlib.pyplot as plt

def reconstruct_image(patches, dictionary, coefficients, image_shape=(150, 150)):
    reconstructed_patches = np.dot(coefficients, dictionary.T)
    reconstructed_image = np.zeros(image_shape)
    patch_idx = 0
    for i in range(0, image_shape[0] - 8 + 1, 4):
        for j in range(0, image_shape[1] - 8 + 1, 4):
            reconstructed_image[i:i + 8, j:j + 8] = reconstructed_patches[patch_idx].reshape(8, 8)
            patch_idx += 1
    
    return reconstructed_image

def visualize_dictionary(dictionary):
    fig, axes = plt.subplots(1, 10, figsize=(15, 15))
    for i in range(10):
        ax = axes[i]
        ax.imshow(dictionary[:, i].reshape(8, 8), cmap='gray')
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    dictionary = np.load('../outputs/dictionary.npy')
    visualize_dictionary(dictionary)
