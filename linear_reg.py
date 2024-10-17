# Import the necessary libraries
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data (X is the input, y is the output)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction
prediction = model.predict([[6]])
print(f"Prediction for input 6: {prediction}")

