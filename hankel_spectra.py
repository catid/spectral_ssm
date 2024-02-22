import os
import numpy as np
from scipy.sparse import linalg

# Function to generate the Hankel matrix introduced in "Learning Linear Dynamical Systems via Spectral Filtering" (Hazan, 2017)
def optimized_hankel_matrix_2017(L):
    i, j = np.indices((L, L)) + 1  # Adjust for 1-based indexing

    # Compute H[i, j] using vectorized operations
    H = 2 / ((i + j) ** 3 - (i + j))
    return H

# Function to generate the Hankel matrix introduced in the appendix of "Spectral State Space Models" (Agarwal, 2024)
def optimized_hankel_matrix_2024(L):
    i, j = np.indices((L, L)) + 1  # Adjust for 1-based indexing

    # Compute H[i, j] using vectorized operations
    numerator = ((-1) ** (i + j - 2) + 1) * 8
    denominator = (i + j + 3) * (i + j - 1) * (i + j + 1)
    H = numerator / denominator
    return H

# Function to compute and return truncated spectral decomposition
def truncated_spectral_decomp(A, k=32):
    L = A.shape[0]
    assert A.shape[0] == A.shape[1], "Must be square"

    eigenvals, eigenvecs = np.linalg.eigh(A)

    return eigenvals[-k:], eigenvecs[:, -k:]

def load_or_compute_eigen_data(L=256, K=16, matrix_version="2024", cache_file="hankel_spectra.npz"):
    eigenvals_key = f'eigenvals_{matrix_version}_L_{L}'
    eigenvecs_key = f'eigenvecs_{matrix_version}_L_{L}'

    cache_exists = os.path.exists(cache_file)
    if cache_exists:
        with np.load(cache_file) as data:
            if eigenvals_key in data and eigenvecs_key in data:
                # Load precomputed eigenvalues and eigenvectors from the cache
                eigenvals = data[eigenvals_key]
                eigenvecs = data[eigenvecs_key]
                print(f"Loaded precomputed eigenvalues and eigenvectors for version {matrix_version}, L={L} from cache.")
                return eigenvals[:K], eigenvecs[:, :K]

    # If the data is not available in the cache, compute it
    if matrix_version == '2017':
        matrix = optimized_hankel_matrix_2017(L)
    elif matrix_version == '2024':
        matrix = optimized_hankel_matrix_2024(L)
    else:
        raise ValueError("Invalid matrix version. Choose '2017' or '2024'.")

    eigenvals, eigenvecs = truncated_spectral_decomp(matrix)

    # Update the cache with the newly computed values
    all_data = {}
    if cache_exists:
        with np.load(cache_file) as data:
            all_data.update(dict(data))
    all_data[eigenvals_key] = eigenvals
    all_data[eigenvecs_key] = eigenvecs
    np.savez(cache_file, **all_data)

    print(f"Computed and cached eigenvalues and eigenvectors for version {matrix_version}, L={L}.")
    return eigenvals[:K], eigenvecs[:, :K]
