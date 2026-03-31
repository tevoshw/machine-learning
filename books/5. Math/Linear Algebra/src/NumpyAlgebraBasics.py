"""
Linear Algebra with NumPy
Vectors, Matrices, and Tensors - Creation and Basic Operations
"""

import numpy as np

print("=" * 80)
print("PART 1: CREATION OF VECTORS, MATRICES, AND TENSORS")
print("=" * 80)

# ============================================================================
# VECTORS (1D arrays)
# ============================================================================
print("\n--- VECTORS (1D) ---\n")

vector = np.array([1, 2, 3])
print(f"Vector:\n{vector}")
print(f"Shape: {vector.shape}")  # (3,)
print(f"Dimensions: {vector.ndim}")  # 1
print(f"Size (number of elements): {vector.size}")  # 3
print(f"Data type: {vector.dtype}\n")  # int64


# ============================================================================
# MATRICES (2D arrays)
# ============================================================================
print("\n--- MATRICES (2D) ---\n")

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")  # (2, 3) → 2 rows, 3 columns
print(f"Dimensions: {matrix.ndim}")  # 2
print(f"Size (number of elements): {matrix.size}")  # 6
print(f"Data type: {matrix.dtype}\n")  # int64


# ============================================================================
# TENSORS (3D arrays and beyond)
# ============================================================================
print("\n--- TENSORS (3D) ---\n")

tensor = np.array([[[1, 2],
                    [3, 4]],
                   [[5, 6],
                    [7, 8]]])
print(f"Tensor:\n{tensor}")
print(f"Shape: {tensor.shape}")  # (2, 2, 2) → 2 depth, 2 rows, 2 columns
print(f"Dimensions: {tensor.ndim}")  # 3
print(f"Size (number of elements): {tensor.size}\n")  # 8


# ============================================================================
# TENSORS (4D - Batch of Images)
# ============================================================================
print("\n--- TENSORS (4D - Batch of Images) ---\n")

# Common in deep learning: (batch_size, height, width, channels)
batch_images = np.random.rand(32, 224, 224, 3)  # 32 RGB images of 224x224
print(f"Shape: {batch_images.shape}")  # (32, 224, 224, 3)
print(f"Dimensions: {batch_images.ndim}")  # 4
print(f"Size (number of elements): {batch_images.size}\n")


print("\n" + "=" * 80)
print("PART 2: CREATING MATRICES WITH SPECIFIC PATTERNS")
print("=" * 80)

# ============================================================================
# Ones matrix
# ============================================================================
print("\n--- Matrix of Ones ---\n")

matrix_ones = np.ones((2, 3))
print(f"Ones matrix (2x3):\n{matrix_ones}")
print(f"Shape: {matrix_ones.shape}\n")


# ============================================================================
# Zeros matrix
# ============================================================================
print("\n--- Matrix of Zeros ---\n")

matrix_zeros = np.zeros((3, 3))
print(f"Zeros matrix (3x3):\n{matrix_zeros}")
print(f"Shape: {matrix_zeros.shape}\n")


# ============================================================================
# Identity matrix
# ============================================================================
print("\n--- Identity Matrix ---\n")

matrix_identity = np.eye(3)
print(f"Identity matrix (3x3):\n{matrix_identity}")
print(f"Shape: {matrix_identity.shape}\n")


# ============================================================================
# Random matrix
# ============================================================================
print("\n--- Random Matrix ---\n")

matrix_random = np.random.rand(2, 3)  # Random values between 0 and 1
print(f"Random matrix (2x3):\n{matrix_random}")
print(f"Shape: {matrix_random.shape}\n")

matrix_random_int = np.random.randint(1, 10, size=(2, 3))  # Random integers
print(f"Random integer matrix (2x3):\n{matrix_random_int}\n")


print("\n" + "=" * 80)
print("PART 3: CREATING SEQUENCES")
print("=" * 80)

# ============================================================================
# arange: Create array with start, stop, step
# ============================================================================
print("\n--- arange: Start, Stop, Step ---\n")

vector_arange = np.arange(1, 10, 0.5)
print(f"Array from 1 to 10 (step 0.5):\n{vector_arange}")
print(f"Shape: {vector_arange.shape}\n")


# ============================================================================
# linspace: Create array with start, stop, number of elements
# ============================================================================
print("\n--- linspace: Start, Stop, Number of Elements ---\n")

vector_linspace = np.linspace(0, 10, 11)  # 11 equally spaced values from 0 to 10
print(f"Array from 0 to 10 (11 elements):\n{vector_linspace}")
print(f"Shape: {vector_linspace.shape}\n")


print("\n" + "=" * 80)
print("PART 4: RESHAPING")
print("=" * 80)

# ============================================================================
# Reshape: Change shape without changing data
# ============================================================================
print("\n--- Reshape: Change Dimensions ---\n")

matrix_flat = np.array([1, 2, 3, 4, 5, 6])
print(f"Original array:\n{matrix_flat}")
print(f"Original shape: {matrix_flat.shape}\n")

matrix_2x3 = matrix_flat.reshape(2, 3)  # Reshape to 2 rows, 3 columns
print(f"Reshaped to (2x3):\n{matrix_2x3}")
print(f"New shape: {matrix_2x3.shape}\n")

matrix_3x2 = matrix_flat.reshape(3, 2)  # Reshape to 3 rows, 2 columns
print(f"Reshaped to (3x2):\n{matrix_3x2}")
print(f"New shape: {matrix_3x2.shape}\n")

matrix_1x6 = matrix_flat.reshape(1, 6)  # Reshape to row vector
print(f"Reshaped to (1x6):\n{matrix_1x6}")
print(f"New shape: {matrix_1x6.shape}\n")

# Flatten: Convert to 1D
matrix_flattened = matrix_2x3.flatten()
print(f"Flattened from (2x3) to 1D:\n{matrix_flattened}")
print(f"Shape: {matrix_flattened.shape}\n")


print("\n" + "=" * 80)
print("PART 5: CHANGING DATA TYPES (dtype)")
print("=" * 80)

# ============================================================================
# dtype conversion
# ============================================================================
print("\n--- Data Type Conversion ---\n")

matrix_int = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype=np.int32)
print(f"Integer matrix (int32):\n{matrix_int}")
print(f"Data type: {matrix_int.dtype}\n")

matrix_float = matrix_int.astype(np.float64)
print(f"Converted to float64:\n{matrix_float}")
print(f"Data type: {matrix_float.dtype}\n")

matrix_complex = matrix_int.astype(np.complex128)
print(f"Converted to complex128:\n{matrix_complex}")
print(f"Data type: {matrix_complex.dtype}\n")


print("\n" + "=" * 80)
print("PART 6: BASIC OPERATIONS")
print("=" * 80)

# ============================================================================
# Vector operations
# ============================================================================
print("\n--- Vector Operations ---\n")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"v1: {v1}")
print(f"v2: {v2}\n")

print(f"v1 + v2 = {v1 + v2}  (addition)")
print(f"v1 - v2 = {v1 - v2}  (subtraction)")
print(f"v1 * 2 = {v1 * 2}  (scalar multiplication)")
print(f"v1 * v2 = {v1 * v2}  (element-wise multiplication)")
print(f"v1 · v2 = {np.dot(v1, v2)}  (dot product)\n")


# ============================================================================
# Matrix operations
# ============================================================================
print("\n--- Matrix Operations ---\n")

m1 = np.array([[1, 2, 3],
               [4, 5, 6]])
m2 = np.array([[7, 8],
               [9, 10],
               [11, 12]])

print(f"m1 (shape {m1.shape}):\n{m1}\n")
print(f"m2 (shape {m2.shape}):\n{m2}\n")

# Matrix multiplication
result = np.dot(m1, m2)  # or m1 @ m2
print(f"m1 @ m2 (shape {result.shape}):\n{result}")
print(f"Explanation: (2,3) × (3,2) = (2,2)\n")

# Element-wise multiplication (Hadamard product)
m3 = np.array([[1, 2],
               [3, 4]])
m4 = np.array([[5, 6],
               [7, 8]])
print(f"Element-wise multiplication (Hadamard product):")
print(f"m3 * m4 = \n{m3 * m4}\n")

# Matrix transpose
print(f"m1.T (transpose):\n{m1.T}")
print(f"Shape change: {m1.shape} → {m1.T.shape}\n")


print("\n" + "=" * 80)
print("PART 7: USEFUL ATTRIBUTES AND METHODS")
print("=" * 80)

print("\n--- Summary of Attributes ---\n")

test_matrix = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])

print(f"Matrix:\n{test_matrix}\n")
print(f".shape       → {test_matrix.shape}      (rows, columns)")
print(f".ndim        → {test_matrix.ndim}       (number of dimensions)")
print(f".size        → {test_matrix.size}      (total elements)")
print(f".dtype       → {test_matrix.dtype}  (data type)")
print(f".T           → Shape {test_matrix.T.shape}   (transpose)")
print(f".sum()       → {test_matrix.sum()}     (sum all elements)")
print(f".mean()      → {test_matrix.mean():.2f}    (mean all elements)")
print(f".max()       → {test_matrix.max()}      (maximum element)")
print(f".min()       → {test_matrix.min()}      (minimum element)")

print("\n" + "=" * 80)