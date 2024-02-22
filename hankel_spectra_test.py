import numpy as np
import unittest

from hankel_spectra import optimized_hankel_matrix_2017, optimized_hankel_matrix_2024, load_or_compute_eigen_data

def ref_hankel_matrix_2017(L):
    # Initialize the matrix with zeros
    H = [[0 for _ in range(L)] for _ in range(L)]

    # Loop through each element to compute its value
    for i in range(1, L + 1):  # Adjust for 1-based indexing
        for j in range(1, L + 1):  # Adjust for 1-based indexing
            # Compute H[i, j] according to the given formula
            H[i-1][j-1] = 2 / ((i + j) ** 3 - (i + j))

    return H

# Google version for reference
def get_hankel_matrix(
    n: int,
) -> np.ndarray:
  z = np.zeros((n, n))
  for i in range(1, n + 1):
    for j in range(1, n + 1):
      z[i - 1, j - 1] = 2 / ((i + j) ** 3 - (i + j))
  return z

def ref_hankel_matrix_2024(L):
    # Initialize an empty matrix of size LxL
    H = np.zeros((L, L))
    
    # Loop over each element in the matrix
    for i in range(L):  # Adjusting for 0-based indexing in the loop
        for j in range(L):
            # Compute the 1-based indices
            i_1based = i + 1
            j_1based = j + 1
            
            # Compute H[i, j] using the specified formula
            numerator = ((-1) ** (i_1based + j_1based - 2) + 1) * 8
            denominator = (i_1based + j_1based + 3) * (i_1based + j_1based - 1) * (i_1based + j_1based + 1)
            
            H[i, j] = numerator / denominator
    
    return H

# Google version for reference
def get_top_hankel_eigh(
    h: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    eig_vals, eig_vecs = np.linalg.eigh(h)

    return eig_vals[-k:], eig_vecs[:, -k:]

# Explicitly sort to make sure these are sorted
def get_top_hankel_eigh_explicit_sort(h: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    eig_vals, eig_vecs = np.linalg.eigh(h)
    
    # Argsort the eigenvalues in descending order
    idx = eig_vals.argsort()[::-1]  # This gets indices that would sort the array in descending order

    # Sort the eigenvalues and eigenvectors using the indices
    eig_vals_sorted = eig_vals[idx]
    eig_vecs_sorted = eig_vecs[:, idx]

    # Select the top k eigenvalues and corresponding eigenvectors
    top_k_eig_vals = eig_vals_sorted[:k]
    top_k_eig_vecs = eig_vecs_sorted[:, :k]

    # Reverse order
    return top_k_eig_vals[::-1], top_k_eig_vecs[:, ::-1]


class TestHankelMatrices(unittest.TestCase):
    
    def test_hankel_matrix_2017(self):
        """Verify that the optimized 2017 Hankel matrix matches the reference implementation."""
        for L in [8, 512]:
            Z_reference = ref_hankel_matrix_2017(L)
            Z_google = get_hankel_matrix(L)
            Z_optimized = optimized_hankel_matrix_2017(L)
            np.testing.assert_array_equal(Z_reference, Z_optimized, "Optimized 2017 version does not match")
            np.testing.assert_array_equal(Z_optimized, Z_google, "Our 2017 opt version does not match Google version")
            np.testing.assert_array_equal(Z_reference, Z_google, "Our 2017 ref version does not match Google version")

    def test_hankel_matrix_2024(self):
        """Verify that the optimized 2024 Hankel matrix matches the reference implementation."""
        for L in [8, 512]:
            Z_reference = ref_hankel_matrix_2024(L)
            Z_optimized = optimized_hankel_matrix_2024(L)
            np.testing.assert_array_equal(Z_reference, Z_optimized, "Optimized 2024 version does not match")

    def test_eigen_data(self):
        """Check if eigenvalues and eigenvectors are correctly computed for different L values."""
        matrix_versions = ['2024', '2017']
        K = 32
        for matrix_version in matrix_versions:
            for L in [2**i for i in range(3, 15)]:
                print(f"Evaluating L={L} matrix={matrix_version}...")
                with self.subTest(L=L, matrix_version=matrix_version):
                    eigenvals, eigenvecs = load_or_compute_eigen_data(L, K, matrix_version=matrix_version)
                    self.assertIsInstance(eigenvals, np.ndarray, f"Eigenvalues for L={L}, version={matrix_version} are not an ndarray")
                    self.assertIsInstance(eigenvecs, np.ndarray, f"Eigenvectors for L={L}, version={matrix_version} are not an ndarray")

                    if L > 512:
                        continue

                    if matrix_version == '2024':
                        Z_reference = ref_hankel_matrix_2024(L)
                    else:
                        Z_reference = get_hankel_matrix(L)

                    ref_eigenvals, ref_eigenvecs = get_top_hankel_eigh(Z_reference, K)

                    ref2_eigenvals, ref2_eigenvecs = get_top_hankel_eigh_explicit_sort(Z_reference, K)

                    #print(f"L={L} {matrix_version} eigenvals={eigenvals}")
                    #print(f"ref_eigenvals: {ref_eigenvals}")
                    #print(f"delta against ref_eigenvals: {eigenvals - ref_eigenvals}")
                    #print(f"eigenvals.shape = {eigenvals.shape}")
                    #print(f"ref_eigenvals.shape = {ref_eigenvals.shape}")
                    #print(f"eigenvecs.shape = {eigenvecs.shape}")
                    #print(f"ref_eigenvecs.shape = {ref_eigenvecs.shape}")

                    np.testing.assert_array_equal(eigenvals, ref_eigenvals, "eigenvals: Our version does not match Google version")
                    np.testing.assert_array_equal(eigenvecs, ref_eigenvecs, "eigenvecs: Our version does not match Google version")

                    np.testing.assert_array_equal(eigenvals, ref2_eigenvals, "eigenvals: Not sorted")
                    np.testing.assert_array_equal(eigenvecs, ref2_eigenvecs, "eigenvecs: Not sorted")

if __name__ == '__main__':
    unittest.main()
