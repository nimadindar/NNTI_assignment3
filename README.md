# PCA on MNIST Dataset

This project demonstrates the use of **Principal Component Analysis (PCA)** to analyze and reduce dimensionality in the MNIST dataset using PyTorch. The notebook includes data preprocessing, eigendecomposition, variance analysis, and reconstruction error calculations.

## Concepts Implemented

1. **Data Loading and Preprocessing**:
   - The MNIST dataset is loaded using PyTorch’s `torchvision.datasets`.
   - The images are normalized and reshaped to be compatible with PCA. Each image is flattened from a 28x28 grid to a 784-dimensional vector.

2. **Eigendecomposition of the Covariance Matrix**:
   - The PCA method begins by calculating the covariance matrix of the data.
   - Eigendecomposition is applied to extract **eigenvalues** and **eigenvectors** of the covariance matrix:
     - **Eigenvalues** represent the variance explained by each principal component.
     - **Eigenvectors** define the directions of these components.
   - Cumulative variance explained by principal components is calculated to analyze how much variance is retained with a subset of components.

3. **Variance Analysis**:
   - Plots the cumulative variance explained as a function of the number of principal components.
   - Helps in choosing the number of components needed to retain a specified amount of variance, which is critical for dimensionality reduction.

4. **Reconstruction and Error Calculation**:
   - Using a subset of the top principal components, the original images are reconstructed.
   - Calculates the **PCA reconstruction error** to quantify how much information is lost when using a limited number of components.
   - The reconstruction error is computed as the normalized Frobenius norm of the difference between the original and reconstructed data.

## Folder Structure

- `data/`: Contains the MNIST dataset (automatically downloaded by the script).
- `src/`: Code implementation files:
  - `eigen_decomposition.py`: Performs eigendecomposition on the covariance matrix.
  - `variance_analysis.py`: Plots cumulative variance explained by principal components.
  - `reconstruction.py`: Reconstructs images using selected principal components and computes reconstruction error.

## Usage Example

The following steps outline how to load the dataset, perform PCA, and analyze results:

```python
import torch
from torchvision import datasets, transforms
from src.eigen_decomposition import eigen_decomp
from src.reconstruction import pca_reconstruction_error

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
data = mnist_data.data[:2000].float() / 255.0
X = data.view(data.size(0), -1)  # Flatten images to 784 dimensions

# Perform PCA to get eigenvalues, eigenvectors, and cumulative variance
eigvals, eigvecs, cumulative_variance = eigen_decomp(X)

# Calculate reconstruction error using the top p components
error, reconstructed_data = pca_reconstruction_error(X, p=50, eigvecs=eigvecs)
print(f"Reconstruction error with 50 components: {error}")
```
Requirements
Python 3.x
PyTorch
Matplotlib (for plotting)
torchvision (for dataset loading)
