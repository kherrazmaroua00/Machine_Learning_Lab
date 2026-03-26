# logistic_regression_ready.py

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# -------- Step 1: Define the dataset --------
X = [
    [2200, 15], [2750, 20], [5000, 40], [4000, 20], [3300, 20],
    [2000, 10], [2500, 12], [12000, 80], [2880, 10], [2300, 15],
    [1500, 10], [3000, 18], [2000, 10], [2150, 8], [3400, 20],
    [5000, 20], [4000, 10], [3300, 15], [2000, 12], [2500, 14],
    [10000, 100], [3150, 10], [2950, 15], [1500, 5], [3000, 18],
    [8000, 12], [2220, 14], [6000, 100], [3050, 10], [2000, 14]
]

y = [
    1,1,0,0,1,1,1,1,0,1,
    1,0,1,1,0,1,0,0,0,1,
    1,1,0,1,0,1,0,1,1,0
]

# Convert to numpy arrays for safety
X = np.array(X)
y = np.array(y)

# -------- Step 2: Standardize features --------
ss = StandardScaler()
X_train = ss.fit_transform(X)

print("Standardized X_train:\n", X_train)

# -------- Step 3: Train Logistic Regression --------
lr = LogisticRegression()
lr.fit(X_train, y)

# -------- Step 4: Predict a new sample --------
testX = [[500, 2]]
X_test = ss.transform(testX)

label = lr.predict(X_test)
prob = lr.predict_proba(X_test)

print("\nValue to be predicted:", X_test)
print("Predicted label =", label)
print("Probability =", prob)