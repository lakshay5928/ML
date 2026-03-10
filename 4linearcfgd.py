import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# dataset 
data = { 
"X": [1, 2, 3, 4, 5], 
"y": [2, 4, 5, 4, 5] 
} 
df = pd.DataFrame(data) 
X = df["X"].values 
y = df["y"].values 
m = len(y)   # number of samples 
# feature scaling 
X = (X - np.mean(X)) / np.std(X) 
# Initialize Parameters 
theta0 = 0    # intercept 
theta1 = 0    # slope 
learning_rate = 0.1 
epochs = 1000 
 
# Define Hypothesis Function 
def hypothesis(theta0, theta1, X): 
    return theta0 + theta1 * X 
 
# Cost Function (Mean Squared Error) 
def compute_cost(theta0, theta1, X, y): 
    predictions = hypothesis(theta0, theta1, X) 
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2) 
 
# Gradient Descent Algorithm 
cost_history = [] 
 
for _ in range(epochs): 
    predictions = hypothesis(theta0, theta1, X) 
     
    d_theta0 = (1 / m) * np.sum(predictions - y) 
    d_theta1 = (1 / m) * np.sum((predictions - y) * X) 
     
    theta0 -= learning_rate * d_theta0 
    theta1 -= learning_rate * d_theta1 
     
    cost = compute_cost(theta0, theta1, X, y) 
    cost_history.append(cost) 
 
print(f"Final Intercept (θ0): {theta0:.4f}") 
print(f"Final Slope (θ1): {theta1:.4f}") 
 
# Plot Cost vs Iterations 
plt.plot(range(epochs), cost_history) 
plt.xlabel("Iterations") 
plt.ylabel("Cost (J)") 
plt.title("Cost Function Convergence") 
plt.show() 
plt.scatter(X,y,label='Actual Data') 
plt.plot(X,hypothesis(theta0,theta1,X),color='red',label="Regression Line") 
plt.xlabel('X') 
plt.ylabel('y') 
plt.legend() 
plt.show()