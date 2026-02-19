# Math Foundations: Linear Algebra & Data Structures

In Machine Learning, data is not just a collection of numbers; it's a hierarchy of structures. Understanding the **Rank** and **Shape** of your data is critical to building functional models.

## 1. The Data Hierarchy (Tensors)

| Term | Rank | Description | Analogous to | Python/NumPy Shape |
| :--- | :---: | :--- | :--- | :--- |
| **Scalar** | 0 | A single number | A point | `()` |
| **Vector** | 1 | A list of numbers | A line | `(n,)` |
| **Matrix** | 2 | A grid (Rows x Cols) | A spreadsheet | `(m, n)` |
| **Tensor** | 3+ | A multidimensional array | A cube or hypercube | `(d, m, n)` |

## 2. Deep Dive

### Scalars ($x \in \mathbb{R}$)
The simplest unit. Represents magnitude without direction.
* **In ML:** Used for hyperparameters (Learning Rate: 0.001) or single metrics (Final Loss).

### Vectors ($\mathbf{v} = [x_1, x_2, \dots, x_n]$)
An ordered list of scalars.
* **In ML:** Represents **one single observation**. If you are predicting house prices, your vector might be `[size, rooms, age]`.


### Matrices ($A \in \mathbb{R}^{m \times n}$)
A 2D array where each row is a vector.
* **In ML:** Represents a **Dataset** or a **Batch**.
* **Rows ($m$):** Number of samples.
* **Columns ($n$):** Number of features.

### Tensors
Any structure with rank higher than 2.
* **Example (Images):** A single RGB image is a Rank-3 tensor with shape `(height, width, 3)`.
* **Example (Video/Batch):** A batch of images is Rank-4: `(batch_size, height, width, channels)`.

## 3. The Golden Rule of Matrix Multiplication
For the operation $\mathbf{C = A \cdot B}$ to be valid:
The number of **columns** in the first matrix must match the number of **rows** in the second.

$$(M \times N) \cdot (N \times P) = (M \times P)$$

> **Visual Tip:** The "inner" dimensions must match, and the "outer" dimensions determine the shape of the result.

## 4. Key Operations in ML
1. **Transposition ($A^T$):** Flipping rows into columns. Often used to align shapes before multiplication.
2. **Dot Product:** The core of how neurons calculate signals ($W \cdot X$).
3. **Element-wise Multiplication (Hadamard Product):** Multiplying corresponding elements (used in some activation functions).
