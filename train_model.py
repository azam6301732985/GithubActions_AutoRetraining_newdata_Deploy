import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Load old or new data
if os.path.exists("data/train_new.csv"):
    df = pd.read_csv("data/train_new.csv")
    print("Training on NEW data")
else:
    df = pd.read_csv("data/train_old.csv")
    print("Training on OLD data")

X = df[["feature1", "feature2"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate and save accuracy
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

with open("model/accuracy.txt", "w") as f:
    f.write(f"MSE: {mse}\n")

print(f"Model trained. MSE: {mse}")
