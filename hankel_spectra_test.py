import numpy as np

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

# Verification function
def verify_hankel_matrix_equality(L):
    Z_reference = optimized_hankel_matrix_2017(L)
    Z_optimized = ref_hankel_matrix_2017(L)
    assert np.array_equal(Z_reference, Z_optimized), "Optimized 2017 version does not match"

    Z_reference = optimized_hankel_matrix_2024(L)
    Z_optimized = ref_hankel_matrix_2024(L)
    assert np.array_equal(Z_reference, Z_optimized), "Optimized 2024 version does not match"

# Precompute powers of two
matrix_version = '2017'
k = 32
powers_of_two = [2**i for i in range(8, 15)]

for L in powers_of_two:
    eigenvals, eigenvecs = load_or_compute_eigen_data(L, k, matrix_version="2017")
    print(f"L={L} 2017 eigenvals={eigenvals}")
    eigenvals, eigenvecs = load_or_compute_eigen_data(L, k, matrix_version="2024")
    print(f"L={L} 2024 eigenvals={eigenvals}")

print("Success")
