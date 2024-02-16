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

class TestHankelMatrices(unittest.TestCase):
    
    def test_hankel_matrix_2017(self):
        """Verify that the optimized 2017 Hankel matrix matches the reference implementation."""
        for L in [256, 512]:
            Z_reference = optimized_hankel_matrix_2017(L)
            Z_optimized = ref_hankel_matrix_2017(L)
            np.testing.assert_array_equal(Z_reference, Z_optimized, "Optimized 2017 version does not match")
    
    def test_hankel_matrix_2024(self):
        """Verify that the optimized 2024 Hankel matrix matches the reference implementation."""
        for L in [256, 512]:
            Z_reference = optimized_hankel_matrix_2024(L)
            Z_optimized = ref_hankel_matrix_2024(L)
            np.testing.assert_array_equal(Z_reference, Z_optimized, "Optimized 2024 version does not match")

    def test_eigen_data(self):
        """Check if eigenvalues and eigenvectors are correctly computed for different L values."""
        matrix_versions = ['2017', '2024']
        K = 32
        for matrix_version in matrix_versions:
            for L in [2**i for i in range(8, 15)]:
                with self.subTest(L=L, matrix_version=matrix_version):
                    eigenvals, eigenvecs = load_or_compute_eigen_data(L, K, matrix_version=matrix_version)
                    self.assertIsInstance(eigenvals, np.ndarray, f"Eigenvalues for L={L}, version={matrix_version} are not an ndarray")
                    self.assertIsInstance(eigenvecs, np.ndarray, f"Eigenvectors for L={L}, version={matrix_version} are not an ndarray")
                    # Add more specific tests for eigenvals and eigenvecs here if needed
                    print(f"L={L} {matrix_version} eigenvals={eigenvals}")

if __name__ == '__main__':
    unittest.main()
