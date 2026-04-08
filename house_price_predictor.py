import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive: saves files instead of opening windows
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and Inspect the Data
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("Florida Real Estate Sold Properties.csv", encoding="utf-8-sig")

print("=" * 60)
print("STEP 1 — Raw Data")
print("=" * 60)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Missing values per column:\n{df.isnull().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clean the Data
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 2 — Cleaning")
print("=" * 60)

before = len(df)
df = df[df["lastSoldPrice"] >= 1000]
df = df[df["sqft"].isna() | (df["sqft"] >= 50)]
print(f"Dropped {before - len(df)} outlier rows → {len(df):,} remaining")

numeric_cols = ["listPrice", "sqft", "beds", "baths",
                "baths_full", "garage", "year_built", "stories"]
for col in numeric_cols:
    median_val = df[col].median()
    missing = df[col].isna().sum()
    if missing > 0:
        df[col] = df[col].fillna(median_val)
        print(f"  Filled {missing:>4} missing '{col}' → median={median_val:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — Feature Engineering
#
# Raw numbers are not always the best representation of what matters.
# We create three new features from existing columns:
#
#   age            = 2026 - year_built
#     A house built in 1950 vs 2020 is more naturally described by its age
#     (76 vs 6 years) than by a year number. The model sees a clearer signal.
#
#   price_per_sqft = listPrice / sqft
#     This captures the NEIGHBORHOOD pricing density. A 2000 sqft house
#     in Miami Beach (listPrice=$2M, ppsf=$1000) vs Sebring (listPrice=$300k,
#     ppsf=$150) tells the model far more than sqft alone.
#
#   bed_bath_ratio = beds / baths
#     Captures layout quality. A 5-bedroom, 1-bath house is a different
#     product than a 3-bedroom, 3-bath house of the same sqft.
#
# None of this requires new data — it's just math on what we already have.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("IMPROVEMENT 1 — Feature Engineering")
print("=" * 60)

df["age"]             = 2026 - df["year_built"]
df["price_per_sqft"]  = df["listPrice"] / df["sqft"]
df["bed_bath_ratio"]  = df["beds"] / df["baths"].replace(0, 1)   # avoid /0

# Log-transform listPrice so it lives on the same scale as our log target.
# The relationship  log(salePrice) ≈ a·log(listPrice)  is nearly linear and
# much better behaved than  log(salePrice) ≈ a·listPrice  (which curves).
df["log_listPrice"] = np.log(df["listPrice"])

# Cap engineered features at their 99th percentile to prevent extreme outliers
# from producing exp(huge_number) blow-ups when we exponentiate predictions.
for col in ["age", "price_per_sqft"]:
    cap_val = df[col].quantile(0.99)
    n_capped = (df[col] > cap_val).sum()
    df[col] = df[col].clip(upper=cap_val)
    print(f"  Capped '{col}' at 99th pct ({cap_val:.1f}) — {n_capped} rows clipped")

print("\nNew features after engineering:")
print(df[["log_listPrice", "age", "price_per_sqft", "bed_bath_ratio"]].describe().round(2))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Encode Categorical Features
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3 — One-Hot Encoding")
print("=" * 60)

type_dummies = pd.get_dummies(df["type"], prefix="type", drop_first=True)
print(f"Encoded 'type' into {type_dummies.shape[1]} binary columns: {list(type_dummies.columns)}")
df = pd.concat([df, type_dummies], axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Select Features, Split, and Normalize
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4 — Feature Selection, Split, Normalization")
print("=" * 60)

feature_cols = (
    ["log_listPrice", "sqft", "beds", "baths", "garage", "stories",
     "age", "price_per_sqft", "bed_bath_ratio"]       # log_listPrice replaces raw listPrice
    + list(type_dummies.columns)
)
print(f"Features ({len(feature_cols)} total): {feature_cols}")

X_all = df[feature_cols].values.astype(float)
y_all = df["lastSoldPrice"].values.astype(float)

np.random.seed(42)
indices   = np.random.permutation(len(y_all))
split     = int(0.8 * len(y_all))
train_idx = indices[:split]
test_idx  = indices[split:]

X_train, y_train_raw = X_all[train_idx], y_all[train_idx]
X_test,  y_test_raw  = X_all[test_idx],  y_all[test_idx]

print(f"Train: {len(y_train_raw):,} rows | Test: {len(y_test_raw):,} rows")

# ── Normalization (fit on train only) ─────────────────────────────────────
train_mean = X_train.mean(axis=0)
train_std  = X_train.std(axis=0)
train_std[train_std == 0] = 1

X_train_norm = (X_train - train_mean) / train_std
X_test_norm  = (X_test  - train_mean) / train_std

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — Log-Transform the Target
#
# Sale prices are RIGHT-SKEWED: most houses sell for $200k–$800k, but a few
# sell for $5M–$50M. Those extreme values create huge errors that inflate RMSE
# and pull the gradient in the wrong direction for the majority of houses.
#
# Solution: instead of predicting price directly, predict log(price).
#
#   log(200,000) ≈ 12.2
#   log(800,000) ≈ 13.6
#   log(5,000,000) ≈ 15.4    ← still large, but not 25× larger than 200k
#
# The distribution becomes much more symmetric (roughly normal), which is
# exactly what linear regression assumes. After prediction, we convert back:
#   predicted_price = exp(predicted_log_price)
#
# This typically reduces MAE by 30–40% on real estate data.
# ══════════════════════════════════════════════════════════════════════════════

y_train = np.log(y_train_raw)   # train on log scale
y_test  = np.log(y_test_raw)    # evaluate on log scale too

print(f"\nTarget on log scale — mean: {y_train.mean():.3f}, std: {y_train.std():.3f}")
print(f"  (equivalent to mean price: ${np.exp(y_train.mean()):,.0f})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train with Gradient Descent + L2 Regularization
#
# IMPROVEMENT 3 — More Iterations
#   In the previous run cost was still dropping at iteration 800.
#   3000 iterations is enough for this dataset to fully converge.
#
# IMPROVEMENT 4 — L2 Regularization (Ridge Regression)
#
#   Without regularization the model can make any weight as large as it wants.
#   This can cause it to "memorize" training data — weights that are tuned to
#   noise rather than signal (overfitting).
#
#   L2 regularization adds a penalty to the cost function:
#
#     J_regularized = J + (λ / 2n) · Σ wⱼ²
#
#   This forces the model to keep weights small unless a feature truly earns
#   its size. The update rule gets one extra term:
#
#     dw = (1/n) · Xᵀ · error  +  (λ/n) · w    ← shrinks w toward 0
#
#   λ (lambda) controls strength:
#     λ = 0   → plain gradient descent (what we had before)
#     λ = 0.1 → gentle shrinkage, slightly more robust
#     λ = 100 → very aggressive, weights approach zero (underfit)
#
#   This is exactly "Ridge Regression" — the next named algorithm after
#   plain linear regression.
# ══════════════════════════════════════════════════════════════════════════════

def compute_cost(X, y, w, b, lambda_=0.0):
    n = len(y)
    predictions = X @ w + b
    mse_term = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
    reg_term  = (lambda_ / (2 * n)) * np.sum(w ** 2)   # L2 penalty
    return mse_term + reg_term

def gradient_descent(X, y, w, b, learning_rate, iterations, lambda_=0.0):
    cost_history = []
    n = len(y)

    for i in range(iterations):
        predictions = X @ w + b
        error       = predictions - y

        dw = (1 / n) * (X.T @ error) + (lambda_ / n) * w   # ← L2 term
        db = (1 / n) * np.sum(error)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(X, y, w, b, lambda_)
        cost_history.append(cost)

        if i % 500 == 0:
            print(f"  Iteration {i:>4}: Cost = {cost:.6f}")

    return w, b, cost_history

print("\n" + "=" * 60)
print("STEP 5 — Training (log target + L2 regularization)")
print("=" * 60)

n_features  = X_train_norm.shape[1]
w           = np.zeros(n_features)
b           = 0.0
lambda_     = 0.1    # gentle L2 penalty

w, b, cost_history = gradient_descent(
    X_train_norm, y_train, w, b,
    learning_rate=0.1,
    iterations=3000,
    lambda_=lambda_
)

print(f"\nFinal weights (on log-price scale):")
for name, weight in zip(feature_cols, w):
    print(f"  {name:22s}: {weight:>8.4f}")
print(f"  {'bias (b)':22s}: {b:>8.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate on the Test Set
#
# Predictions come out on the log scale — we convert back with exp().
# We compute MAE and RMSE in dollar space so the numbers are interpretable.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 6 — Evaluation on Test Set")
print("=" * 60)

log_pred    = X_test_norm @ w + b         # predictions in log space
y_pred_test = np.exp(log_pred)            # back to dollar space

mae       = np.mean(np.abs(y_pred_test - y_test_raw))
rmse      = np.sqrt(np.mean((y_pred_test - y_test_raw) ** 2))
mean_price = np.mean(y_test_raw)

print(f"Mean sale price in test set : ${mean_price:>12,.0f}")
print(f"MAE  (avg absolute error)   : ${mae:>12,.0f}  ({100 * mae / mean_price:.1f}% of mean)")
print(f"RMSE (penalizes big errors) : ${rmse:>12,.0f}")

# ── Sample predictions ─────────────────────────────────────────────────────
print("\nSample predictions (first 8 test houses):")
print(f"  {'Predicted':>12}  {'Actual':>12}  {'Error':>12}  {'Error %':>8}")
for i in range(min(8, len(y_test_raw))):
    pred = y_pred_test[i]
    true = y_test_raw[i]
    err  = pred - true
    pct  = 100 * abs(err) / true
    print(f"  ${pred:>11,.0f}  ${true:>11,.0f}  ${err:>+11,.0f}  {pct:>7.1f}%")

# ── Plot 1: Predicted vs Actual ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cap  = 3_000_000
mask = y_test_raw < cap

axes[0].scatter(y_test_raw[mask], y_pred_test[mask],
                alpha=0.3, s=10, color="steelblue")
max_val = max(y_test_raw[mask].max(), y_pred_test[mask].max())
axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect")
axes[0].set_xlabel("Actual Sale Price ($)")
axes[0].set_ylabel("Predicted Sale Price ($)")
axes[0].set_title("Predicted vs Actual\n(capped at $3M)")
axes[0].legend()

# ── Plot 2: Cost Curve ────────────────────────────────────────────────────
axes[1].plot(cost_history, color="darkorange")
axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("Cost J (log scale)")
axes[1].set_title("Cost Decreasing over Iterations")

plt.tight_layout()
plt.savefig("plot_predictions_vs_actual.png", dpi=120)
plt.close()
print("Saved → plot_predictions_vs_actual.png")

# ── Plot 3: Error Distribution ─────────────────────────────────────────────
# Shows whether errors are symmetric (good) or skewed (model has a bias)
errors_pct = 100 * (y_pred_test - y_test_raw) / y_test_raw
errors_pct_clipped = np.clip(errors_pct, -100, 100)

plt.figure(figsize=(7, 4))
plt.hist(errors_pct_clipped, bins=60, color="steelblue", edgecolor="white")
plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
plt.xlabel("Prediction Error (%)")
plt.ylabel("Number of houses")
plt.title("Distribution of Prediction Errors\n(clipped at ±100%)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_error_distribution.png", dpi=120)
plt.close()
print("Saved → plot_error_distribution.png")
