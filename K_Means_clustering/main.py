import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# -------- Step 1: Generate dataset --------
X, y_true = make_blobs(n_samples=50, n_features=2, centers=3, random_state=42)

# -------- Step 2: Apply K-means --------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_pred = kmeans.fit_predict(X)  # التسمية اللي اكتشفها K-means
centroids = kmeans.cluster_centers_  # إحداثيات المراكز

# -------- Step 3: Visualize --------
colors = ['red', 'green', 'blue']
plt.figure(figsize=(6,5))

for i in range(n_clusters):
    plt.scatter(X[y_pred==i, 0], X[y_pred==i, 1], c=colors[i], label=f'Cluster {i}', s=50, edgecolor='k')

# عرض المراكز
plt.scatter(centroids[:,0], centroids[:,1], c='black', marker='X', s=100, label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering Example')
plt.legend()
plt.show()