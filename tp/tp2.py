import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Bluck

# Load the Excel file
file_path = r"C:\Users\HP\Downloads\Prix-Moyen-Au-m²-Algerie.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

list_x = df.iloc[:, 0].tolist()  # Input values
list_y = df.iloc[:, 1].tolist()  # Output values

# Sample data: x and y values
x = np.array(list_x)  # Input values
y = np.array(list_y)  # Output values


print(len(list_x))
# Number of data points
n = len(x)
print(n)

# Calculate the necessary sums
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x_squared = np.sum(x ** 2)

# Calculate a and b using the least squares method
a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
b = (sum_y - a * sum_x) / n


# Make predictions
predictions = a * x + b

# Calculate Mean Squared Error (MSE)
mse = np.mean((y - predictions) ** 2)
print(f'Mean Squared Error = {mse}')

# Output the results
print(f'y= {a} x + {b}')

# Plotting the results
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, predictions, color='red', label='Fitted Line')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid()
plt.show()
