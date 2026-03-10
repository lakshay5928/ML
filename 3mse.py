import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,5,4,5])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)

print("Mean Squared Error(MSE):",mse)
plt.scatter(X,y,color='blue',label='Actual Data')
plt.plot(X,model.predict(X),color='red',label='Linear Regression')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()