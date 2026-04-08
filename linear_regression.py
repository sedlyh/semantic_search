import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([5, 7, 9, 11, 13], dtype=float)

# Cost function
def compute_cost(x, y, m, b):
    n = len(x)
    predictions = m * x + b
    cost = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient descent
def gradient_descent(x, y, m, b, learning_rate, iterations):
    n = len(x)
    cost_history = []

    for i in range(iterations):
        predictions = m * x + b
        
        # Derivatives of cost function
        dm = (1 / n) * np.sum((predictions - y) * x)
        db = (1 / n) * np.sum(predictions - y)

        # Update m and b
        m = m - learning_rate * dm
        b = b - learning_rate * db

        # Save cost for tracking
        cost = compute_cost(x, y, m, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}, m = {m:.6f}, b = {b:.6f}")

    return m, b, cost_history

# Initial values
m = 0
b = 0
learning_rate = 0.01
iterations = 10000

# Train model
m, b, cost_history = gradient_descent(x, y, m, b, learning_rate, iterations)

print("\nFinal values:")
print("m =", m)
print("b =", b)

# Predictions
y_pred = m * x + b

# Plot data and regression line
plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, label="Regression Line")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression from Scratch")
plt.show()

# Plot cost over time
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Decreasing")
plt.show()