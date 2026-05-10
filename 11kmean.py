import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv('kmeans_data.csv')

# Select Features
X = df[['Age', 'Income']]

# Remove Missing Values
X = X.dropna()

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Create KMeans Model
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

# Train Model
kmeans.fit(X_scaled)

# Add Cluster Labels
df['Cluster'] = kmeans.labels_

# Show Dataset
print(df.head())

# Scatter Plot
plt.figure(figsize=(6,4))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=kmeans.labels_,
    cmap='viridis'
)

# Cluster Centers
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    c='red',
    marker='X'
)

plt.title("K-Means Clustering")

plt.xlabel("Age")
plt.ylabel("Income")

plt.show()

# Elbow Method
inertia = []

for k in range(1, 10):

    km = KMeans(
        n_clusters=k,
        random_state=42
    )

    km.fit(X_scaled)

    inertia.append(km.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(6,4))

plt.plot(
    range(1,10),
    inertia,
    marker='o'
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.show()