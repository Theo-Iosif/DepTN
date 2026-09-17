## ===== HELPER FUNCTIONS ===== ##

# --- imports --- #

import numpy as np


# --- helper functions for embeddings and tensor cores --- #

#   -> helper function: Mezzadri's algorithm for generating random unitary matrices

def generate_haar_isometric(rows: int, cols: int):
    '''
    Generate a random unitary matrix of size (rows, cols) using Mezzadri's algorithm.

    input: rows (int): The number of rows in the matrix.
           cols (int): The number of columns in the matrix.

    output: (np.ndarray): A Haar random unitary matrix of size (rows, cols).
    '''
    z = np.random.randn(rows, cols)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)

    return q * (d / np.abs(d))