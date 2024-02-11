import os
import numpy as np
from scipy.sparse import linalg

# Function to generate the Hankel matrix introduced in "Learning Linear Dynamical Systems via Spectral Filtering" (Hazan, 2017)
def generate_hankel_matrix_2017(L):
    i, j = np.indices((L, L))
    sum_indices = i + j + 1
    Z = 2 / (sum_indices ** 3) - sum_indices
    return Z

# Function to generate the Hankel matrix introduced in the appendix of "Spectral State Space Models" (Agarwal, 2024)
def generate_hankel_matrix_2024(L):
    max_sum = 2 * (L - 1)
    ij_values = np.arange(2, max_sum + 1)
    values = ((-1) ** (ij_values - 2) + 1) * 8 / ((ij_values + 3) * (ij_values - 1) * (ij_values + 1))
    Z = np.zeros((L, L))
    i, j = np.indices(Z.shape)
    sum_indices = i + j
    Z = np.where(sum_indices >= 2, values[sum_indices - 2], 0)
    return Z

# Function to compute and return truncated spectral decomposition
def truncated_spectral_decomp(A, k):
    L = A.shape[0]
    assert A.shape[0] == A.shape[1], "Must be square"
    assert k < L, "k must be less than L"

    if k >= L:
        eigenvals, eigenvecs = np.linalg.eigh(A)
        # Sort eigenvalues and eigenvectors
    else:
        eigenvals, eigenvecs = linalg.eigsh(
            A=A,
            k=k,
            which="LM",
            return_eigenvectors=True,
        )

    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    return eigenvals, eigenvecs.T

def load_or_compute_eigen_data(L=256, k=16, matrix_version="2024", cache_file="eigenvalues_eigenvectors.npz"):
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
                return eigenvals, eigenvecs

    # If the data is not available in the cache, compute it
    if matrix_version == '2017':
        matrix = generate_hankel_matrix_2017(L)
    elif matrix_version == '2024':
        matrix = generate_hankel_matrix_2024(L)
    else:
        raise ValueError("Invalid matrix version. Choose '2017' or '2024'.")

    eigenvals, eigenvecs = truncated_spectral_decomp(matrix, k)

    # Update the cache with the newly computed values
    all_data = {}
    if cache_exists:
        with np.load(cache_file) as data:
            all_data.update(dict(data))
    all_data[eigenvals_key] = eigenvals
    all_data[eigenvecs_key] = eigenvecs
    np.savez(cache_file, **all_data)

    print(f"Computed and cached eigenvalues and eigenvectors for version {matrix_version}, L={L}.")
    return eigenvals, eigenvecs

# Example of loading or computing eigenvalues and eigenvectors
matrix_version = '2017'

k = 32
powers_of_two = [2**i for i in range(8, 15)]
for L in powers_of_two:
    eigenvals, eigenvecs = load_or_compute_eigen_data(L, k, matrix_version="2017")
    print(f"L={L} 2017 eigenvals={eigenvals}")
    eigenvals, eigenvecs = load_or_compute_eigen_data(L, k, matrix_version="2024")
    print(f"L={L} 2024 eigenvals={eigenvals}")
