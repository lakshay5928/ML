import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("ML/Salary.csv")

print("Dataset:\n", df)

# Select two rows
x1, y1 = df.loc[0, ["YearsExperience", "Salary"]]
x2, y2 = df.loc[1, ["YearsExperience", "Salary"]]

# -----------------------------
# 1 Euclidean Distance
# -----------------------------
euclidean = math.sqrt((x1-x2)**2 + (y1-y2)**2)

print("\nEuclidean Distance:", euclidean)

# -----------------------------
# 2 Manhattan Distance
# -----------------------------
manhattan = abs(x1-x2) + abs(y1-y2)

print("Manhattan Distance:", manhattan)

# -----------------------------
# 3 Minkowski Distance
# -----------------------------
p = 3

minkowski = ((abs(x1-x2)**p) + (abs(y1-y2)**p))**(1/p)

print("Minkowski Distance:", minkowski)

# -----------------------------
# 4 Cosine Similarity
# -----------------------------
vector1 = df.loc[0, ["YearsExperience","Salary"]].values.reshape(1,-1)
vector2 = df.loc[1, ["YearsExperience","Salary"]].values.reshape(1,-1)

cosine = cosine_similarity(vector1, vector2)

print("Cosine Similarity:", cosine[0][0])
