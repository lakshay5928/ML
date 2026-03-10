import numpy as np 
import matplotlib.pyplot as plt 

np.random.seed(42) 
m=100 
X= np.random.randn(m,2) 
y=(X[:,0]+X[:,1]>0).astype(int) 
X=np.c_[np.ones(m),X]
 
def sigmoid(z): 
 return 1/(1+np.exp(-z))
 
def compute_cost(X,y,theta): 
 m=len(y) 
 h=sigmoid(X @ theta) 
 cost=(-1/m) * np.sum(y*np.log(h + 1e-9) + (1-y)*np.log(1-h + 1e-9)) 
 return cost
 
def compute_gradient(X,y,theta): 
 m=len(y) 
 h=sigmoid(X @ theta) 
 gradient= (1/m) * X.T @ (h-y) 
 return gradient
 
def gradient_descent(X,y,theta, learning_rate,iterations): 
 cost_history=[] 
 for i in range(iterations): 
  gradient=compute_gradient(X,y,theta) 
  theta=theta-learning_rate*gradient 
  cost=compute_cost(X,y,theta) 
  cost_history.append(cost) 
  if i%100==0: 
   print(f"Iteration {i}: Cost {cost:.4f}") 
 return theta,cost_history
 
theta_initial= np.zeros(X.shape[1]) 
learning_rate=0.01 
iterations=1000 
theta_final,cost_history=gradient_descent(X,y,theta_initial,learning_rate,iterations) 
print("\nFinal Parameters:",theta_final) 

plt.plot(cost_history) 
plt.xlabel("Iterations") 
plt.ylabel("Cost(Log Loss)") 
plt.title("Cost Function Convergence") 
plt.show() 

def predict(X,theta): 
 probabilities=sigmoid(X @ theta) 
 return (probabilities>=0.5).astype(int) 
 
y_pred=predict(X,theta_final) 
accuracy=np.mean(y_pred==y) 
print(f"Training Accuracy: {accuracy*100:.2f}%")