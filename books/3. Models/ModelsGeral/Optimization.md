# OPTIMIZATION METHODS
**DEFINITION**
- Math Objective: Find the better numbers for parameters to decrease the loss function
- Transform and adjust the parameters to improve the predictions


# Optimization Methods: Fundamentals

## 1. GRADIENT (The Directional Guide)
* **Classification:** Mathematical Concept / Vector.
* **Technical Definition:** It is the vector of partial derivatives of the loss function ($Loss$) with respect to the parameters. It points in the direction where the error **increases** fastest.
* **Role in the Model:** Acts as a compass. The optimizer calculates the gradient to know in which direction to adjust the weights (always in the opposite direction to the gradient to decrease the error).
* **Insight:** The gradient determines the **direction** of the update.

## 2. LEARNING RATE / LR (The Intensity Regulator)
* **Classification:** Hyperparameter (manually defined).
* **Technical Definition:** A scalar ($\eta$) that multiplies the gradient, defining the magnitude of the weight update.
* **Update Equation:** $$\theta_{new} = \theta_{old} - \text{LR} \cdot \nabla J(\theta)$$
* **Role in the Model:** Determines the **step size** (speed).
    * **LR too high:** Can cause instability and the model may never converge (skip the ideal point).
    * **LR too low:** Training becomes extremely slow and may get stuck in local minima (holes in the path).

## 3. CONVERGENCE (The Destination State)
* **Classification:** Concept / Final Objective.
* **Definition:** The state reached when the loss function reaches a minimum value and stabilizes.
* **Success Signal:** When the **Gradient** approaches zero and updates via **LR** no longer significantly alter the results.


## Connection Summary (The Logic of the Motor)
| Component | Function | Analogy |
| :--- | :--- | :--- |
| **Gradient** | Direction | The compass indicating descent. |
| **Learning Rate** | Magnitude | The size of the step you take. |
| **Parameters** | Adjustment | The screws being tightened. |
| **Convergence** | Goal | Reaching the bottom of the valley (minimum error). |

## GRADIENT-BASED METHODS

## ADAPTIVES METHODS

## SECOND ORDER METHODS

## METHODS WITHOUT GRADIENT

## CONVEX METHODS / MATH PROGAMMING 