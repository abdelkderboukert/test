import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the Excel file
file_path = r"C:\Users\HP\Downloads\Prix-Moyen-Au-m²-Algerie.xlsx"
df = pd.read_excel(file_path)
df.dropna(inplace=True)

# Extract input and output values
list_x = df.iloc[:, 0].tolist()  # Input values
list_y = df.iloc[:, 1].tolist()  # Output values

# Convert lists to numpy arrays
x = np.array(list_x).reshape(-1, 1)  # Input values reshaped for sklearn
y = np.array(list_y)  # Output values

# Step 2: Apply linear regression
model = LinearRegression()
model.fit(x, y)

# Coefficients of the regression
a = model.coef_[0]
b = model.intercept_
print(f"Coefficients of the regression: a = {a}, b = {b}")

# Step 3: Plot the data and the regression line
y_pred = model.predict(x)

# Step 4: Implement the cost function
def compute_cost(x, y, a, b):
    m = len(y)
    predictions = a * x + b
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

# Step 5: Apply gradient descent
def gradient_descent(x, y, a, b, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = a * x + b
        # Calculate gradients
        a_gradient = (1/m) * np.dot(x.T, (predictions - y))
        b_gradient = (1/m) * np.sum(predictions - y)
        # Update coefficients
        a -= learning_rate * a_gradient
        b -= learning_rate * b_gradient
    return a, b

# Initialize parameters
a_init = 0.0
b_init = 0.0
learning_rate = 0.01
iterations = 1000

# Apply gradient descent
a_final, b_final = gradient_descent(x, y, a_init, b_init, learning_rate, iterations)
print(f"Coefficients after gradient descent: a = {a_final}, b = {b_final}")

def predict_price(surface_area):
    surface_area = np.array(surface_area).reshape(-1, 1)  # Reshape for prediction
    predicted_price = model.predict(surface_area)
    return predicted_price

try:
    input_surface_area = float(input("Enter the surface area in m²: "))  # Prompt user for input
    predicted_price = predict_price([input_surface_area])
    print(f"Predicted price for surface area {input_surface_area} m²: {predicted_price[0]}")
except ValueError:
    print("Please enter a valid number for the surface area.")

# Calculate final cost
cost = compute_cost(x, y, a_final, b_final)
print(f"Final cost: {cost}")

# Function to predict y for a given x
