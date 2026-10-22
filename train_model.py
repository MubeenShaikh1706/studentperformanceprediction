
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH =  "student-mat.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

# 1. Load the dataset
df = pd.read_csv(DATA_PATH, sep=";")

# 2. Select input features and target
features = ["studytime", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model on unseen test data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model training completed!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# 6. Save the trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
