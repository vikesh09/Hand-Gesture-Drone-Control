import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

data = pd.read_csv("hand_data.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "gesture_model.pkl")
print("Model saved as gesture_model.pkl")
