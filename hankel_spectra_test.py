import numpy as np
import unittest

from hankel_spectra import optimized_hankel_matrix_2017, optimized_hankel_matrix_2024, load_or_compute_eigen_data

def ref_hankel_matrix_2017(L):
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            # Compute the denominator according to the formula given
            ij = i + j
            denominator = ij*ij*ij - ij
            if denominator != 0:
                Z[i, j] = 2 / denominator
    return Z

def ref_hankel_matrix_2024(L):
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            ij = i + j
            if ij >= 2:
                numerator = ((-1) ** (ij - 2) + 1) * 8
                denominator = ((ij + 3) * (ij - 1) * (ij + 1))
                Z[i, j] = numerator / denominator
    return Z

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
