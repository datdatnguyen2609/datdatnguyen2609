import numpy as np
import matplotlib.pyplot as plt

# Tao du lieu 2D don gian de minh hoa
def create_synthetic_data(n_per_cluster=50, random_state=0):
    rng = np.random.RandomState(random_state)

    # Tao 3 cum diem
    mean1 = [0, 0]
    mean2 = [5, 5]
    mean3 = [0, 5]

    cov = [[0.5, 0.0],
           [0.0, 0.5]]

    X1 = rng.multivariate_normal(mean1, cov, n_per_cluster)
    X2 = rng.multivariate_normal(mean2, cov, n_per_cluster)
    X3 = rng.multivariate_normal(mean3, cov, n_per_cluster)

    X = np.vstack([X1, X2, X3])
    return X

# K Means tu cai dat don gian, co luu lai cac buoc de ve hinh
def kmeans_with_history(X, K=3, max_iter=10, random_state=0):
    rng = np.random.RandomState(random_state)
    N, D = X.shape

    # Chon K diem ngau nhien lam tam cum ban dau
    init_idx = rng.choice(N, K, replace=False)
    centroids = X[init_idx].copy()

    history = []  # Luu (centroids, labels) qua tung vong lap

    labels = np.zeros(N, dtype=int)

    for it in range(max_iter):
        # Buoc 1: Gan moi diem vao cum gan nhat
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)

        # Luu lai buoc hien tai (truoc khi cap nhat centroid)
        history.append((centroids.copy(), labels.copy()))

        # Buoc 2: Cap nhat tam cum
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # Neu cum rong thi chon lai ngau nhien
                new_centroids[k] = X[rng.randint(0, N)]

        # Kiem tra hoi tu
        shift = np.sqrt(np.sum((centroids - new_centroids) ** 2))
        centroids = new_centroids

        if shift < 1e-4:
            # Luu buoc cuoi cung
            history.append((centroids.copy(), labels.copy()))
            break

    return centroids, labels, history

# Ve qua trinh K Means theo tung buoc
def plot_kmeans_history(X, history, save_path="kmeans_demo.png"):
    num_steps = len(history)
    # Lay toi da 4 buoc de ve cho gon
    steps_to_show = min(4, num_steps)

    plt.figure(figsize=(12, 3))

    for i in range(steps_to_show):
        centroids, labels = history[i]

        plt.subplot(1, steps_to_show, i + 1)
        # Ve cac diem theo mau cua nhan cum
        for k in range(centroids.shape[0]):
            mask = (labels == k)
            plt.scatter(X[mask, 0], X[mask, 1], s=10, alpha=0.7, label=f"Cluster {k}")

        # Ve tam cum
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    s=100, marker="X", edgecolors="black", linewidths=1.0,
                    label="Centroid")

        plt.title(f"Buoc {i + 1}")
        plt.xlabel("x1")
        plt.ylabel("x2")

        # Chi hien legend o subplot cuoi cho dep
        if i == steps_to_show - 1:
            plt.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Da luu hinh minh hoa K Means tai: {save_path}")

def main():
    # 1. Tao du lieu
    X = create_synthetic_data(n_per_cluster=60, random_state=42)

    # 2. Chay K Means va ghi lai lich su
    centroids, labels, history = kmeans_with_history(
        X,
        K=3,
        max_iter=10,
        random_state=1
    )

    # 3. Ve qua trinh hoi tu cua K Means
    plot_kmeans_history(X, history, save_path="kmeans_demo.png")

if __name__ == "__main__":
    main()
