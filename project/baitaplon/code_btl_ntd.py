"""
Nguyen Thanh Dat - 211440519 - BTL
FER2013 K-Means Clustering - HOG + PCA (core version)
- HOG tu cai dat
- K-Means tu cai dat
- PCA dung thu vien
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ==================== CAU HINH ====================
DATA_DIR = "fer2013"
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
N_CLUSTERS = 7
IMG_SIZE = (48, 48)

HOG_CELL_SIZE = 6
HOG_BLOCK_SIZE = 2
HOG_BINS = 9
PCA_COMPONENTS = 0.95

# ==================== 1. LOAD DU LIEU ====================
def load_data(data_folder, max_images_per_emotion=500):
    print(f"\nDang load du lieu tu {data_folder}...")
    
    images = []
    labels = []
    
    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(data_folder, emotion)
        if not os.path.exists(emotion_path):
            print(f"Khong tim thay: {emotion_path}")
            continue
        
        files = os.listdir(emotion_path)[:max_images_per_emotion]
        for img_file in files:
            img_path = os.path.join(emotion_path, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype(np.float32) / 255.0
                if img.std() < 0.02:
                    continue
                images.append(img.flatten())
                labels.append(idx)
            except Exception:
                continue
        
        print(f"  {emotion}: {sum(np.array(labels)==idx)} anh")
    
    images = np.array(images)
    labels = np.array(labels)
    print(f"\nTong so anh: {images.shape[0]}")
    print(f"Input shape: {images.shape}")
    return images, labels

# ==================== 2. HOG TU CAI DAT ====================
def hog_single_image(img,
                     cell_size=HOG_CELL_SIZE,
                     block_size=HOG_BLOCK_SIZE,
                     nbins=HOG_BINS):
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180.0

    h, w = img.shape
    cell_h = cell_w = cell_size
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    hist_cells = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    bin_size = 180.0 / nbins

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y_start = cy * cell_h
            y_end = y_start + cell_h
            x_start = cx * cell_w
            x_end = x_start + cell_w

            cell_mag = mag[y_start:y_end, x_start:x_end].reshape(-1)
            cell_angle = angle[y_start:y_end, x_start:x_end].reshape(-1)

            for m, a in zip(cell_mag, cell_angle):
                bin_idx = int(a // bin_size)
                if bin_idx == nbins:
                    bin_idx = nbins - 1
                hist_cells[cy, cx, bin_idx] += m

    block_h = block_w = block_size
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1

    eps = 1e-6
    hog_vector = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = hist_cells[by:by+block_h, bx:bx+block_w, :].ravel()
            norm = np.sqrt(np.sum(block ** 2) + eps)
            block = block / norm
            hog_vector.extend(block)

    return np.array(hog_vector, dtype=np.float32)

def extract_hog_features(images_flat, img_size=IMG_SIZE):
    print("\nTrich dac trung HOG tu cai dat...")
    n_samples = images_flat.shape[0]
    hog_features = []

    for i in range(n_samples):
        img = images_flat[i].reshape(img_size)
        feat = hog_single_image(img)
        hog_features.append(feat)
        if (i + 1) % 500 == 0:
            print(f"  Da xu ly {i+1} anh")

    hog_features = np.array(hog_features)
    print(f"HOG feature shape: {hog_features.shape}")
    return hog_features

# ==================== 3. K-MEANS TU CAI DAT ====================
def kmeans_single_run(X, n_clusters, max_iter=300, random_state=None):
    N, D = X.shape
    rng = np.random.RandomState(random_state)

    init_idx = rng.choice(N, n_clusters, replace=False)
    centroids = X[init_idx].copy()

    for _ in range(max_iter):
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)

        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = X[rng.randint(0, N)]

        shift = np.sqrt(np.sum((centroids - new_centroids) ** 2))
        centroids = new_centroids
        if shift < 1e-4:
            break

    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    closest = np.min(dist_sq, axis=1)
    inertia = np.sum(closest)

    return labels, centroids, inertia

def apply_kmeans(features,
                 n_clusters=N_CLUSTERS,
                 n_init=10,
                 max_iter=300,
                 base_random_state=42):
    print(f"\nChay K-Means tu cai dat, K = {n_clusters}...")
    best_inertia = None
    best_labels = None
    best_centroids = None

    for i in range(n_init):
        rstate = base_random_state + i
        labels, centroids, inertia = kmeans_single_run(
            features,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=rstate
        )
        if (best_inertia is None) or (inertia < best_inertia):
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    print("Hoan thanh K-Means")
    return best_centroids, best_labels

# ==================== 4. MAP CLUSTER SANG LABEL ====================
def map_clusters_to_emotions(cluster_labels, true_labels):
    print("\nMap cluster sang emotion pho bien nhat...")
    cluster_map = {}
    for cluster_id in range(N_CLUSTERS):
        mask = (cluster_labels == cluster_id)
        emotions_in_cluster = true_labels[mask]
        if len(emotions_in_cluster) > 0:
            most_common = Counter(emotions_in_cluster).most_common(1)[0][0]
            cluster_map[cluster_id] = most_common
            print(f"  Cluster {cluster_id} -> {EMOTIONS[most_common]}")
        else:
            print(f"  Cluster {cluster_id} rong")
    mapped_labels = np.array([cluster_map.get(c, -1) for c in cluster_labels])
    return cluster_map, mapped_labels

# ==================== 5. DANH GIA ====================
def evaluate(features, cluster_labels, true_labels, mapped_labels):
    print("\n========== DANH GIA ==========")
    try:
        silhouette = silhouette_score(features, cluster_labels)
        print(f"Silhouette Score: {silhouette:.4f}")
    except Exception as e:
        silhouette = float('nan')
        print("Khong tinh duoc Silhouette:", e)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    acc = accuracy_score(true_labels, mapped_labels)

    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"Accuracy sau mapping: {acc:.4f}")

    cm = confusion_matrix(true_labels, mapped_labels)
    print("Confusion matrix:")
    print(cm)

    return {
        "silhouette": silhouette,
        "ari": ari,
        "nmi": nmi,
        "accuracy": acc,
        "confusion_matrix": cm
    }

# ==================== MAIN ====================
def main():
    print("\n==========================================")
    print("FER2013 K-MEANS CLUSTERING - CORE VERSION")
    print("==========================================")

    train_path = os.path.join(DATA_DIR, "training")
    images, labels = load_data(train_path, max_images_per_emotion=300)

    if images.size == 0:
        print("Khong co du lieu, thoat")
        return

    print(f"\nTong so anh: {len(images)}")

    # HOG
    features = extract_hog_features(images)

    # StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Da chuan hoa feature bang StandardScaler")

    # PCA
    print(f"\nAp dung PCA, input dim = {features.shape[1]}")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    features = pca.fit_transform(features)
    print(f"Output dim sau PCA: {features.shape[1]}")
    print(f"Tong ti le phuong sai: {pca.explained_variance_ratio_.sum():.4f}")

    # KMeans
    if features.shape[0] < N_CLUSTERS:
        print("So mau < so cluster, khong chay duoc KMeans")
        return

    centroids, cluster_labels = apply_kmeans(features)

    # Map cluster sang emotion
    cluster_map, mapped_labels = map_clusters_to_emotions(cluster_labels, labels)

    # Danh gia
    metrics = evaluate(features, cluster_labels, labels, mapped_labels)

    print("\n=========== TONG KET ===========")
    print(f"Silhouette: {metrics['silhouette']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"ARI:        {metrics['ari']:.4f}")
    print(f"NMI:        {metrics['nmi']:.4f}")
    print("================================")

if __name__ == "__main__":
    main()
