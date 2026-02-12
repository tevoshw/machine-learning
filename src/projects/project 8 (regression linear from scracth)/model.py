import numpy as np

# w1 (Study) ≈ 0.85
# w2 (Sleep)   ≈ 0.25
X = np.array([
    [1.0, 7.0], [2.5, 8.0], [4.0, 7.5], [5.5, 6.0], [6.0, 8.0],
    [7.5, 7.0], [8.0, 5.0], [8.5, 9.0], [9.0, 6.5], [10.0, 7.5]])

y = np.array([
    [3.5], [5.0], [6.5], [7.0], [8.2], 
    [8.8], [8.5], [9.8], [9.5], [10.0]])

class Model:
    def __init__(self, X, Y):
        # GET THE DATA
        self.X = X
        self.Y = Y

        # INICIALIZE THE PARAMETERS AND HYPER
        self.weights = np.random.randn(2, 1)
        self.iteration = len(X)
        self.n_parameters = len(self.weights)
        self.lr = 0.001 

    def forward_pass(self, x):
        # GET THE Y'
        return (self.weights[0] * self.X[x][0]) + (self.weights[1] * self.X[x][1]) 
    
    def loss_function(self, yu, y_idx):
        # MSE 
        return (yu - self.Y[y_idx]) ** 2
    
    def backpropagation(self, yu, y_idx):
        # CALCULATE THE DERIVATES FOR EACH WEIGHT
        gradient = []
        for x in range(0, self.n_parameters):
            derivate_loss = 2 * (yu - self.Y[y_idx])
            derivate_weight = self.X[y_idx][x]

            gradient.append(derivate_loss * derivate_weight)

        return gradient
    
    def optimization(self, gradient_vector):
        # UPDATE THE PARAMETERS
        for x in range(0, len(gradient_vector)):
            self.weights[x] = self.weights[x] - (self.lr * gradient_vector[x])
    
    def fit_model(self):
        # TRAINS THE MODEL
        for x in range(0, self.iteration):
            yu = self.forward_pass(x)
            loss = self.loss_function(yu, x)
            print(f"A loss atual é de: {loss}")
            
            backprop = self.backpropagation(yu, x)
            self.optimization(backprop)
            print(f"Pesos atualizados: w1:{self.weights[0]}, w2:{self.weights[1]}")

model = Model(X, y)
model.fit_model()