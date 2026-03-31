
import numpy as np


matriz1 = np.array([ [1,2,3],
                     [4,5,6] ])

print(f"Matriz 1:\n{matriz1}") # See the matriz
print(f"Matriz 1 shape: {matriz1.shape}")  # See the shape of the matriz (rows, columns)
print(f"Matriz 1 number of dimensions: {matriz1.ndim}") # See the number of dimensions of the matriz (2 in this case, because it's a 2D array)
print(f"Matriz 1 size (numbers of elements): {matriz1.size}") # See the size of the matriz (number of elements, in this case 6)
print(f"Matriz 1 data type: {matriz1.dtype}") # See the data type of the matriz (int64 in this case, because it's an array of integers)


# Change the dtype of the matriz to float64
matriz2 = np.array([ [7,8,9],
                     [10,11,12] ], dtype=np.float64)
print(f"\nMatriz 2:\n{matriz2}")
print(f"Matriz 2 shape: {matriz2.dtype}")

# Create a matriz only with 1 numebrs n,m
matriz3 = np.ones((2,2))
print(f"\nMatriz 3 (ones):\n{matriz3}")

# Create a matriz only with 0 numebrs n,m
matriz4 = np.zeros((3,3))
print(f"\nMatriz 4 (zeros):\n{matriz4}")

# Create an array to 1 from 10, in a step of 0.5
vector1 = np.arange(1,10, 0.5)
print(f"\nVector 1:\n{vector1}")

# Change the shape of the matriz
matriz5 = np.array([1,2,3,4,5,6])  
print(f"\nMatriz 5:\n{matriz5}")
matriz5 = matriz5.reshape(2,3) # Will become a matriz with 2 rows and 3 columns (n,m)
print(f"Matriz 5 reshaped to 2x3:\n{matriz5}")

#