import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

# Show first 5 rows
print(df.head())

# Shape
print("Shape of the dataset:", df.shape)

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# Remove null values
df = df.dropna()

# Show available columns
print("\nColumns in Dataset:")
print(df.columns)

# Matplotlib Scatter Plot
plt.figure(figsize=(6,4))

plt.scatter(df['Age'], df['Cholesterol'])

plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Age vs Cholesterol')

plt.show()

# Seaborn Scatter Plot
plt.figure(figsize=(6,4))

sns.scatterplot(
    x='Age',
    y='Cholesterol',
    data=df
)

plt.title('Age vs Cholesterol Using Seaborn')

plt.xlabel('Age')
plt.ylabel('Cholesterol')

plt.show()