import numpy as np
import matplotlib.pyplot as plt

# ── Dataset ───────────────────────────────────────────────────────────────────
# Each row is one training example with 2 features:
#   x1 = house size (hundreds of sq ft)
#   x2 = number of bedrooms
# Target: y = price (thousands of $)
# True relationship: y = 3*x1 + 10*x2 + 5

X = np.array([
    [1, 1, 10, 2],
    [2, 1, 12, 1],
    [3, 2, 14, 3],
    [4, 3, 16, 2],
    [5, 3, 18, 3],
    [6, 4, 20, 4],
    [7, 4, 22, 4],
    [8, 5, 24, 5],
], dtype=float)

y = np.array([18, 21, 34, 47, 50, 63, 66, 79], dtype=float)

# ── Feature Normalization ──────────────────────────────────────────────────────
# Without this, features on different scales cause gradient descent to take
# very uneven steps, slowing convergence dramatically.
def normalize(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    return (X - mean) / std, mean, std

X_norm, X_mean, X_std = normalize(X)

# ── Cost Function (MSE) ────────────────────────────────────────────────────────
# J(w, b) = (1 / 2n) * Σ (ŷᵢ - yᵢ)²
# Identical in shape to the single-feature version; X·w replaces m*x.
def compute_cost(X, y, w, b):
    n = len(y)
    predictions = X @ w + b          # shape: (n,)
    cost = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
    return cost

# ── Gradient Descent ──────────────────────────────────────────────────────────
# Partial derivatives:
#   ∂J/∂wⱼ = (1/n) · Σ (ŷᵢ - yᵢ) · xᵢⱼ   → compact: (1/n) · Xᵀ · (ŷ - y)
#   ∂J/∂b  = (1/n) · Σ (ŷᵢ - yᵢ)
def gradient_descent(X, y, w, b, learning_rate, iterations):
    n = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X @ w + b                        # (n,)
        error       = predictions - y                  # (n,)

        dw = (1 / n) * (X.T @ error)                  # (features,)
        db = (1 / n) * np.sum(error)                  # scalar

        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 200 == 0:
            print(f"Iteration {i:>4}: Cost = {cost:.6f} | w = {np.round(w, 4)} | b = {b:.4f}")

    return w, b, cost_history

# ── Training ───────────────────────────────────────────────────────────────────
n_features    = X_norm.shape[1]
w             = np.zeros(n_features)   # one weight per feature
b             = 0.0
learning_rate = 0.1
iterations    = 1000

print("Training multi-feature linear regression...\n")

w, b, cost_history = gradient_descent(X_norm, y, w, b, learning_rate, iterations)

# ── Results ────────────────────────────────────────────────────────────────────
print("\nFinal parameters (on normalized features):")
print(f"  w = {w}")
print(f"  b = {b:.4f}")

# ── Predict on training data ───────────────────────────────────────────────────
y_pred = X_norm @ w + b

print("\nPredictions vs actual:")
for i in range(len(y)):
    print(f"  x={X[i]} → predicted: {y_pred[i]:.2f}, actual: {y[i]:.2f}")

# ── Predict a new house ────────────────────────────────────────────────────────
new_house = np.array([5.5, 3.5])
new_house_norm = (new_house - X_mean) / X_std
predicted_price = new_house_norm @ w + b
print(f"\nNew house {new_house} → predicted price: {predicted_price:.2f}k")

# ── Plot 1: Predictions vs Actual ─────────────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(y,      "o-", label="Actual",    color="steelblue")
plt.plot(y_pred, "s--", label="Predicted", color="tomato")
plt.xticks(range(len(y)), [str(list(x.astype(int))) for x in X], rotation=15)
plt.xlabel("Training example [x1, x2]")
plt.ylabel("Price (k$)")
plt.title("Multi-Feature Linear Regression — Predictions vs Actual")
plt.legend()
plt.tight_layout()
plt.show()

# ── Plot 2: Cost Curve ─────────────────────────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(cost_history, color="darkorange")
plt.xlabel("Iterations")
plt.ylabel("Cost J")
plt.title("Cost Function Decreasing over Iterations")
plt.tight_layout()
plt.show()
