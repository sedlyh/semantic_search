# Machine Learning from Scratch — Florida House Price Predictor

This project teaches machine learning by building a real house price predictor
from scratch using only math and Python — no ML libraries like scikit-learn.
Every concept is introduced only when it is needed to solve a real problem.

---

## What is Machine Learning?

Machine learning is teaching a computer to find patterns in data by example,
instead of writing explicit rules.

Instead of writing:

```
if sqft > 2000 and beds >= 3 and zip == "33446":
    price = 650000
```

You show the computer thousands of real house sales and let it figure out the
pattern itself. The computer learns a mathematical formula that maps inputs
(size, bedrooms, location) to an output (price).

---

## The Goal

Given information about a Florida house — its size, number of bedrooms,
bathrooms, garage, age, property type — **predict what it will sell for**.

---

## Files in This Project

| File | What it does |
|---|---|
| `linear_regression.py` | Simplest possible version: one feature, one line |
| `linear_regression_multifeature.py` | Extended to multiple features |
| `house_price_predictor.py` | Full real-world predictor on 10,893 Florida sales |
| `Florida Real Estate Sold Properties.csv` | The dataset |
| `plot_predictions_vs_actual.png` | Output: scatter plot and cost curve |
| `plot_error_distribution.png` | Output: histogram of prediction errors |

---

## How to Run

```bash
pip install numpy pandas matplotlib
python3 house_price_predictor.py
```

---

## The Algorithm: Linear Regression with Gradient Descent

### What is a line?

A line through data has this equation:

```
predicted_price = m × size + b
```

- `m` = slope (how much price increases per square foot)
- `b` = intercept (the base price before size is considered)

With multiple features (size, bedrooms, bathrooms...):

```
predicted_price = w1×size + w2×beds + w3×baths + ... + b
```

- `w1, w2, w3...` are called **weights** — one per feature
- `b` is called the **bias**

The model learns the right values for all the weights by looking at data.

### What is Gradient Descent?

Gradient descent is how the model learns.

Imagine you are blindfolded on a hilly landscape and you want to reach the
lowest valley. You can only feel the slope under your feet. Your strategy:

1. Feel which direction is downhill
2. Take a small step in that direction
3. Repeat until you stop going down

The "landscape" is the **cost function** — a measure of how wrong the model
is right now. Gradient descent navigates it to find the lowest point (the
parameters that make the fewest errors).

---

## Step-by-Step Walkthrough

### Step 1 — Load the Data

```python
df = pd.read_csv("Florida Real Estate Sold Properties.csv", encoding="utf-8-sig")
```

`pandas` loads the CSV into a **DataFrame** — think of it as a smart
spreadsheet in memory. Each row is one house sale. Each column is one property
of that house.

We check:
- How many rows and columns we have (10,893 × 14)
- Which columns have missing values (garage is missing 33% of the time)

---

### Step 2 — Clean the Data

Real data is messy. Two problems to fix:

**A) Outliers** — a house that sold for $10 or has 1 sqft is clearly a data
entry error. Those rows would teach the model wrong things. We remove them:

```python
df = df[df["lastSoldPrice"] >= 1000]
df = df[df["sqft"].isna() | (df["sqft"] >= 50)]
```

**B) Missing values** — you cannot feed `NaN` (empty cell) into a math
formula. We fill each empty cell with the **median** of that column.

Why median and not average?
- Average is pulled by extreme values. If 99 houses have 2 garages and
  1 house has a 16-car garage, the average is inflated.
- Median always gives you the "middle" value — much more stable.

```python
df["garage"] = df["garage"].fillna(df["garage"].median())
```

---

### Step 3 — Feature Engineering

**Feature engineering** means creating new columns from existing ones that
give the model better information.

We create three new features:

| New column | Formula | Why it helps |
|---|---|---|
| `age` | `2026 - year_built` | Age (27 years old) is more intuitive than a year (1999) |
| `price_per_sqft` | `listPrice / sqft` | Captures neighborhood pricing — $150/sqft vs $1,000/sqft |
| `bed_bath_ratio` | `beds / baths` | A 5-bed/1-bath house is a different product than 3-bed/3-bath |
| `log_listPrice` | `log(listPrice)` | Makes the asking price work better on a log scale (explained in Step 6) |

None of this requires new data — just math on what we already have.

We also **cap** `age` and `price_per_sqft` at their 99th percentile.
A property listed for $59,000,000 divided by 100 sqft gives a
price_per_sqft of $590,000 — an extreme outlier that would otherwise
cause the model to make absurd predictions.

---

### Step 4 — One-Hot Encoding

The `type` column contains words: `single_family`, `condos`, `land`, etc.
Math does not understand words. We convert them to numbers.

**One-hot encoding** turns one text column into several binary (0 or 1)
columns:

```
type             →   type_condos   type_land   type_mobile   type_single_family ...
single_family    →   0             0           0             1
condos           →   1             0           0             0
land             →   0             1           0             0
```

Each house gets a 1 in exactly one column and 0 everywhere else.

We drop one of the columns (`drop_first=True`) because it is redundant —
if a house is not condos, not land, not mobile, and not multi-family,
it must be single-family. Keeping the last column would give the model
duplicate information, which breaks the math.

---

### Step 5 — Train/Test Split

Imagine studying for an exam using last year's answers. You would score
perfectly on that exam — but that tells you nothing about whether you
actually understand the material.

The same problem exists in ML. If we measure accuracy on the same data
we trained on, the model can just "memorize" all the answers and look
perfectly accurate while being useless on new houses.

**The solution:** keep 20% of houses hidden during training. After training
is done, test on those houses for the first time. That is your honest score.

```
All 10,884 houses
├── 8,707 houses (80%) → Training set   ← model learns from these
└── 2,177 houses (20%) → Test set       ← model never sees these until evaluation
```

---

### Step 6 — Normalization

Features are on wildly different scales:
- `listPrice` ranges from $10,000 to $59,000,000
- `beds` ranges from 0 to 20

When gradient descent adjusts weights, it takes the same step size for
every feature. If `listPrice` is 3,000,000 times larger than `beds`, the
model takes huge erratic steps for price and tiny useless steps for beds.

**Normalization** rescales every feature to have mean=0 and std=1:

```
normalized = (value - mean) / standard_deviation
```

After normalization all features live roughly between -2 and +2. Gradient
descent now takes balanced steps and converges much faster.

**Critical rule:** compute the mean and standard deviation from the
**training set only**, then apply the same transformation to the test set.
If you compute them from the test set separately, you are secretly letting
the model "peek" at future data.

---

### Step 7 — Log-Transform the Target

House prices are **skewed**: most houses sell for $200k–$800k, but a small
number sell for $5M–$50M. Those extremes distort training because gradient
descent spends disproportionate effort trying to predict rare luxury mansions
at the expense of accuracy on typical houses.

**Solution:** instead of predicting price directly, predict `log(price)`.

```
log(200,000) ≈ 12.2
log(800,000) ≈ 13.6
log(5,000,000) ≈ 15.4     ← still large, but not 25× larger than 200k
```

The log scale compresses the range and makes the distribution symmetric —
which is exactly what linear regression assumes.

After prediction, we convert back with `exp()`:

```python
y_train = np.log(y_train_raw)           # compress to log scale
# ... train the model ...
y_pred = np.exp(X_test_norm @ w + b)    # expand back to dollars
```

Because we now predict `log(price)`, we also need `log(listPrice)` as a
feature — the log-log relationship `log(sale) ≈ a × log(list)` is nearly
linear, while `log(sale) ≈ a × listPrice` is curved and hard to fit.

---

### Step 8 — The Cost Function

The cost function measures how wrong the model is. We use
**Mean Squared Error (MSE)**:

```
J = (1 / 2n) × Σ (predicted - actual)²
```

- Square the errors so positive and negative errors do not cancel out
- A $50k over-prediction and a $50k under-prediction are both bad
- The `2` in the denominator is a math convenience — it cancels cleanly
  when we take the derivative

With L2 regularization (see Step 9), the cost also penalizes large weights:

```
J = MSE + (λ / 2n) × Σ w²
```

This forces the model to keep weights small unless a feature genuinely
deserves to be large.

---

### Step 9 — Gradient Descent + L2 Regularization

The gradient is the direction of "steepest uphill" on the cost surface.
We move in the **opposite** direction (downhill) by a small step each
iteration.

**Partial derivatives** (how much the cost changes if we nudge each weight):

```
∂J/∂w = (1/n) × Xᵀ × (predicted - actual)  +  (λ/n) × w   ← L2 term
∂J/∂b = (1/n) × Σ (predicted - actual)
```

**Update rule** (move downhill):

```python
w = w - learning_rate × dw
b = b - learning_rate × db
```

`learning_rate` controls the step size:
- Too large → the model overshoots the minimum and diverges
- Too small → the model converges but takes thousands of extra iterations

**L2 Regularization** (also called Ridge Regression) adds the `(λ/n) × w`
term to the gradient. This shrinks weights toward zero each step unless the
data strongly supports them. It prevents **overfitting** — when a model
memorizes noise in the training data instead of learning the real pattern.

```
λ = 0    → plain gradient descent
λ = 0.1  → gentle regularization (what we use)
λ = 100  → very aggressive — weights approach zero, model underfits
```

---

### Step 10 — Evaluation

We measure accuracy with two metrics:

**MAE — Mean Absolute Error**
```
MAE = mean( |predicted - actual| )
```
The average dollar amount we are wrong by. Easy to interpret:
"on average, we are off by $23,204."

**RMSE — Root Mean Squared Error**
```
RMSE = sqrt( mean( (predicted - actual)² ) )
```
Squares the errors before averaging, so large mistakes count
disproportionately more. A $200k error counts 4× more than a $100k error.
Use RMSE when very large errors are especially costly.

---

## Results

| Metric | Value |
|---|---|
| Mean sale price | $562,029 |
| MAE | $23,204 (4.1% of mean) |
| RMSE | $97,779 |

This means on an average $562k house, our prediction is off by about $23k —
built entirely from scratch with no ML library.

---

## How Each Improvement Changed the Model

| Improvement | What it solved | MAE before | MAE after |
|---|---|---|---|
| Baseline (v1) | First working model | — | $32,383 (5.8%) |
| Log-transform + log(listPrice) | Skewed price distribution | $32,383 | reduced |
| Feature engineering | Richer information | — | improved |
| L2 regularization | Overfitting, noisy weights | — | cleaner weights |
| Final model | All combined | $32,383 | **$23,204 (4.1%)** |

---

## Key Concepts Glossary

| Term | Plain English |
|---|---|
| Feature | An input column the model learns from (sqft, beds, etc.) |
| Target | The output we want to predict (lastSoldPrice) |
| Weight | How much each feature matters in the prediction |
| Bias | A baseline offset — the price before any features are considered |
| Cost function | A score for how wrong the model is right now |
| Gradient | The direction of steepest increase in cost |
| Gradient descent | Walking downhill on the cost surface to minimize error |
| Learning rate | The size of each step during gradient descent |
| Normalization | Rescaling features to the same scale so gradient descent is balanced |
| Imputation | Filling missing values with a reasonable substitute (e.g. median) |
| One-hot encoding | Converting text categories into binary (0/1) columns |
| Train/test split | Keeping some data hidden to measure honest accuracy |
| Overfitting | Memorizing training data instead of learning the real pattern |
| Regularization | Penalizing large weights to prevent overfitting |
| MAE | Average absolute prediction error in dollars |
| RMSE | Like MAE but large errors count more |
| Log transform | Compressing a skewed distribution to make it more symmetric |

---

## What to Learn Next

You now have the full foundation. These are the natural next steps:

1. **Polynomial features** — add `sqft²` as a feature to capture curves
   in the relationship between size and price
2. **Decision Trees** — a completely different approach: rules instead of weights
3. **Random Forest** — combine hundreds of decision trees for much better accuracy
4. **Neural Networks** — gradient descent on many layers — you already understand
   the core of how they work
