# Dictionary Learning Project

## 📂 Project Description
This project implements **Dictionary Learning** and **Sparse Coding** techniques to reconstruct images using learned sparse representations. The project extracts image patches, learns a dictionary of basis elements, encodes patches using sparse coding, and reconstructs the image using these sparse representations.

### ✨ Key Features
- **Patch Extraction:** Extracts small patches from grayscale images for processing.
- **Dictionary Learning:** Uses scikit-learn's `DictionaryLearning` module to learn a set of basis elements (atoms).
- **Sparse Coding:** Implements **Orthogonal Matching Pursuit (OMP)** for sparse representation of image patches.
- **Image Reconstruction:** Rebuilds the image from the sparse codes and learned dictionary.
- **Visualization:** Displays and saves the original and reconstructed images.

---

## 🛠️ Dependencies
Ensure you have the following packages installed:

```sh
pip install numpy matplotlib scikit-learn Pillow
```

| Package       | Version   | Description                                |
|---------------|------------|--------------------------------------------|
| `numpy`       | >=1.22.0   | For numerical operations and array handling |
| `matplotlib`  | >=3.5.0    | For visualizing original and reconstructed images |
| `scikit-learn`| >=1.1.0    | Provides `DictionaryLearning` and `OMP` modules |
| `Pillow`      | >=9.0.0    | For image loading and processing            |

---

## 🚀 Installation
1. **Clone the Repository:**
```sh
git clone https://github.com/yourusername/dictionary_learning.git
cd dictionary_learning
```

2. **Set up a Virtual Environment:**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
```sh
pip install -r requirements.txt
```

4. **Directory Structure:**
```
dictionary_learning/
├── data/
│   ├── images/                # Contains input images
│   └── patches/               # Stores preprocessed patches as .npy files
├── src/
│   ├── dictionary_learning.py # Main script for learning and reconstruction
│   └── utils.py               # Utility functions (e.g., patch extraction)
├── output/                    # Stores generated images (e.g., reconstructed_image.png)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Usage
1. **Prepare Data:**
   - Place input images in the `data/images` directory.

2. **Run the Dictionary Learning Script:**
```sh
python src/dictionary_learning.py
```

3. **View Results:**
   - The reconstructed image will be saved in the `output` directory as `reconstructed_image.png`.

---

## 🧠 How It Works
1. **Image Loading:** Loads and converts images to grayscale using Pillow.
2. **Patch Extraction:** Extracts small patches from the image using `extract_image_patches` from `utils.py`.
3. **Dictionary Initialization:** Initializes the dictionary randomly.
4. **Dictionary Learning:** Learns a dictionary of basis elements using `DictionaryLearning` from scikit-learn.
5. **Sparse Coding:** Represents patches using a sparse combination of dictionary atoms with `OrthogonalMatchingPursuit`.
6. **Image Reconstruction:** Reconstructs the image by reassembling the patches.
7. **Visualization:** Displays and saves the output.

---

## 🛠️ Troubleshooting
- **Image Not Loaded Error:** Ensure the image path is correct and the file exists in `data/images`.
- **Patch Extraction Issues:** Check if the image is large enough for the specified patch size.
- **Dictionary Learning Convergence:** If the model does not converge, try increasing `n_iter` or preprocessing the patches.

---


