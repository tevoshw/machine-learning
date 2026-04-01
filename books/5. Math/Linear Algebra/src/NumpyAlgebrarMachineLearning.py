import numpy as np

# ForwardPass
''''
Here we created a forward pass method, that we need the:
1. MATRIX (input data) with the (a,b) shape 
2. The WEIGHTS with the (n,m) shape


And their shape should be compatible for the dot product operation (We multiply the columnX from HEIGHTS in every row), here (b == n). The dot product will give us an output with the shape (a,m), which is the output of the forward pass (Z).

'''

    # !------------------------------------------! Create the Matrix !------------------------------------------!
matrix = np.random.randint(1,10, size=(100, 10))
print(f"Input matrix (100x10):\n{matrix}\n")

    # !------------------------------------------! Create the Weights !------------------------------------------!
weights = np.random.rand(10, 1)
print(f"Weights matrix (10x1):\n{weights}\n")


    # !------------------------------------------! Forward Pass (Dot Product) !------------------------------------------!
Z = np.dot(matrix, weights)
print(f"Output of the forward pass (100x1):\n{Z}\n their shape is {Z.shape}")



# Loss Function
""" 

In this part, we will create the target values (the expected output) and then calculate the loss using the Mean Squared Error (MSE) formula. The MSE is calculated as the average of the squared differences between the predicted values (Z) and the target values.
for subtraction operation in matrices, we need to make sure that the shapes of the matrices are compatible. In this case, Z has a shape of (100, 1) and targets also has a shape of (100, 1), so we can directly subtract them without any issues.

"""

    # !------------------------------------------! Create the Target Values !------------------------------------------!
targets = np.random.rand(100, 1)
print(f"Target values (100x1):\n{targets}\n")

    # !------------------------------------------! Calculate the Loss !------------------------------------------!
MSELoss = np.mean((Z - targets) ** 2)
print(f"Mean Squared Error Loss: {MSELoss}\n")

