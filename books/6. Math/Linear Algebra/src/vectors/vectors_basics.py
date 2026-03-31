import numpy as np

# Create the vectors using numpy arrays
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

print("Vector 1:", vector1, "\nVector 2:", vector2)
print("Vector 1 shape:", vector1.shape)
print("Vector 2 shape:", vector2.shape)


# Sum the vectors
vector3 = vector1 + vector2
print("Sum of Vector 1 and Vector 2:", vector3)


# Subtract the vectors
vector4 = vector1 - vector2
print("Difference of Vector 1 and Vector 2:", vector4)

# Dot product of the vectors
"""
The dot product of two vectors isn't their multiplication in the traditional sense. Instead, it's calculated as the sum of the products of their corresponding components. For example, if we have two vectors A = [a1, a2, a3] and B = [b1, b2, b3], their dot product is calculated as:
A . B = a1 * b1 + a2 * b2 + a3 * b3 

"""

vector5 = np.dot(vector1, vector2)
print("Dot product of Vector 1 and Vector 2:", vector5)


# Multiplication of a vector by other vector
vector6 = vector1 * vector2
print("Multiplication of Vector 1 and Vector 2 (element-wise):", vector6)


# Multiplication of a vector by a scalar
scalar = 2
vector7 = scalar * vector1
vector8 = scalar * vector2
print("Multiplication of Vector 1 by scalar 2:", vector7)
print("Multiplication of Vector 2 by scalar 2:", vector8)


# Transpose of a vector
vector9 = np.array([ [1], 
                     [2], 
                     [3]
                        ])  
print("Original Vector 1 (as a column vector):\n", vector9)
vector10 = vector9.T  
print("Transpose of Vector 1 (as a row vector):\n", vector10)