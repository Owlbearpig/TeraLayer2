import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate synthetic 1-dimensional data
data = np.random.randn(100, 1) * 5 + 10

# Visualize the data
plt.scatter(data, np.zeros_like(data), s=30, cmap='viridis', alpha=0.5)
plt.title('Generated 1D Data')
plt.show()

# Reshape the data to be compatible with KMeans
data_reshaped = data.reshape(-1, 1)

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_reshaped)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize the clustered data
plt.scatter(np.zeros_like(data), data, c=labels, s=30, cmap='viridis', alpha=0.5)
plt.scatter(np.zeros_like(centers), centers, marker='X', s=200, color='red', label='Centroids')
plt.title('1D K-means Clustering')
plt.legend()
plt.show()