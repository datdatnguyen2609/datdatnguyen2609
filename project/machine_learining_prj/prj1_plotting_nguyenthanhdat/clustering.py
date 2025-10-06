from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=150, centers=3, random_state=42)

db = DBSCAN(eps=0.3, min_samples=5).fit(X)
plt.scatter(X[:,0], X[:,1], c=db.labels_)
plt.title("DBSCAN Example")
plt.show()  # Dùng # để chú thích, không dùng //
