import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage

# Load Dataset
df = pd.read_csv('kmeans_data.csv')

# Select Features
X = df[['Age', 'Income']]

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Create Agglomerative Clustering Model
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

# Predict Clusters
df['Cluster'] = model.fit_predict(X_scaled)

# Display Dataset
print(df)

# Scatter Plot
plt.figure(figsize=(6,4))

plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=df['Cluster']
)

plt.title("Agglomerative Clustering")

plt.xlabel("Age")
plt.ylabel("Income")

plt.show()

# Create Linkage Matrix
linked = linkage(
    X_scaled,
    method='ward'
)

# Dendrogram
plt.figure(figsize=(8,5))

dendrogram(linked)

plt.title("Dendrogram")

plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()