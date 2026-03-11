# Load dataset once
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("ML/Salary.csv")
df.columns = ["YearsExperience","Salary"]
# -----------------------------
# 1 Euclidean Distance Between Two Data Points
# -----------------------------
x1, y1 = df.loc[0, ["YearsExperience", "Salary"]]
x2, y2 = df.loc[1, ["YearsExperience", "Salary"]]
distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
print("Euclidean Distance:", distance)
# -----------------------------
# 2 Euclidean Distance of Dataset Points from Reference Point
# -----------------------------
x0, y0 = df.loc[0, ["YearsExperience", "Salary"]]
df["Euclidean_Distance"] = np.sqrt(
    (df["YearsExperience"] - x0)**2 +
    (df["Salary"] - y0)**2
)
print(df.head())
# -----------------------------
# 3 Euclidean Distance Between Two Vectors
# -----------------------------
distance_vector = np.sqrt(
    ((df["YearsExperience"] - df["Salary"])**2).sum()
)
print("Euclidean Distance (Vector):", distance_vector)
# -----------------------------
# 4 Manhattan Distance Between Two Points
# -----------------------------
x1, y1 = df.loc[0, ["YearsExperience", "Salary"]]
x2, y2 = df.loc[1, ["YearsExperience", "Salary"]]
manhattan = abs(x1 - x2) + abs(y1 - y2)
print("Manhattan Distance:", manhattan)
# -----------------------------
# 5 Manhattan Distance from Reference Point
# -----------------------------
df["Manhattan_Distance"] = (
    abs(df["YearsExperience"] - x0) +
    abs(df["Salary"] - y0)
)
print(df.head())
# -----------------------------
# 6 Manhattan Distance Between Two Vectors
# -----------------------------
vector_manhattan = abs(df["YearsExperience"] - df["Salary"]).sum()
print("Total Manhattan Distance (Vector):", vector_manhattan)
# -----------------------------
# 7 Cosine Similarity from Reference Point
# -----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["YearsExperience", "Salary"]])
ref = scaled[0]
df["Cosine_Similarity"] = [
    np.dot(ref, row) / (np.linalg.norm(ref) * np.linalg.norm(row))
    for row in scaled
]

print(df.head())


import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("Salary.csv")

# Apply MinMaxScaler
scaler = MinMaxScaler()
df[['YearsExperience','Salary']] = scaler.fit_transform(df[['YearsExperience','Salary']])

# Access two rows using df.loc
x1 = df.loc[0,'YearsExperience']
y1 = df.loc[0,'Salary']

x2 = df.loc[1,'YearsExperience']
y2 = df.loc[1,'Salary']

# 1 Euclidean Distance
euclidean = math.sqrt((x1-x2)**2 + (y1-y2)**2)
print("Euclidean Distance:", euclidean)

# 2 Manhattan Distance
manhattan = abs(x1-x2) + abs(y1-y2)
print("Manhattan Distance:", manhattan)

# 3 Cosine Similarity
cosine = (x1*x2 + y1*y2) / (math.sqrt(x1**2 + y1**2) * math.sqrt(x2**2 + y2**2))
print("Cosine Similarity:", cosine)

# 4 Minkowski Distance (p=3)
p = 3
minkowski = ((abs(x1-x2)**p + abs(y1-y2)**p))**(1/p)
print("Minkowski Distance:", minkowski)
