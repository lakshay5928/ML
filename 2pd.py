import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('ML/Salary.csv')
df.head()

print("Shape of dataset:", df.shape)
print("\nMissing Values: \n",df.isnull().sum())
print("\nStatistical Summary: \n", df.describe())

df=df.dropna()

plt.figure(figsize=(6,4))
plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('YearsExperience vs Salary')
plt.show()

plt.figure(figsize=(6,4))
sns.lineplot(x='YearsExperience',y='Salary',data=df)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Performance ')
plt.show()
