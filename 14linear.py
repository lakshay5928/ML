import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

# Display dataset
print(df.head())

# Shape of dataset
print("Shape of dataset:", df.shape)

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Column names
print("\nColumns:")
print(df.columns)

# Remove null values
df = df.dropna()

# Independent Variable
X = df[['Age']]

# Dependent Variable
y = df['Cholesterol']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Creation
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):",
      mean_absolute_error(y_test, y_pred))

print("R2 Score:",
      r2_score(y_test, y_pred))

# Scatter Plot
plt.figure(figsize=(6,4))

plt.scatter(df['Age'], df['Cholesterol'], color='blue')

plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Age vs Cholesterol')

plt.show()

# Regression Line Plot
plt.figure(figsize=(6,4))

plt.scatter(X_test, y_test, color='blue')

plt.plot(X_test, y_pred, color='red')

plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Linear Regression Line')

plt.show()

# Seaborn Regression Plot
plt.figure(figsize=(6,4))

sns.regplot(
    x='Age',
    y='Cholesterol',
    data=df,
    line_kws={'color':'red'}
)

plt.title('Seaborn Regression Plot')

plt.xlabel('Age')
plt.ylabel('Cholesterol')

plt.show()