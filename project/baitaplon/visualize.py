"""
Nguyen Thanh Dat - 211440519 - BTL
FER2013 K-Means Clustering - HOG + PCA (train/test version)
- HOG tu cai dat
- K-Means tu cai dat
- PCA dung thu vien
- Train tren tap training, danh gia tren tap test
- Ve confusion matrix tren TAP TEST va luu file
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
TRAIN_DIR_NAME = "training"
TEST_DIR_NAME = "test"   # Dat ten theo cau truc folder cua ban

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
N_CLUSTERS = 7
IMG_SIZE = (48, 48)

HOG_CELL_SIZE = 6
HOG_BLOCK_SIZE = 2
HOG_BINS = 9

# Cac moc PCA can thu (ti le phuong sai)
PCA_RATIOS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

FIG_DIR = "figs"

# ==================== 0. HAM VE CONFUSION MATRIX ====================
def plot_confusion_matrix(cm, emotions, title="", save_path=None):
    """
    Ve confusion matrix va luu thanh file anh neu co save_path
    cm la confusion_matrix tren TAP TEST
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    num_classes = len(emotions)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, emotions, rotation=45)
    plt.yticks(tick_marks, emotions)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Da luu confusion matrix: {save_path}")

    plt.close()

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
                # Bo qua anh qua toi, khong co thong tin
                if img.std() < 0.02:
                    continue
                images.append(img.flatten())
                labels.append(idx)
            except Exception:
                continue
        
        print(f"  {emotion}: {sum(np.array(labels) == idx)} anh")
    
    images = np.array(images)
    labels = np.array(labels)
    print(f"\nTong so anh load duoc: {images.shape[0]}")
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

def apply_kmeans_train(features_train,
                       n_clusters=N_CLUSTERS,
                       n_init=10,
                       max_iter=300,
                       base_random_state=42):
    """
    Chay KMeans tren TAP TRAIN
    Tra ve centroids va nhan cluster cho TAP TRAIN
    """
    print(f"\nChay K-Means tu cai dat tren TAP TRAIN, K = {n_clusters}...")
    best_inertia = None
    best_labels = None
    best_centroids = None

    for i in range(n_init):
        rstate = base_random_state + i
        labels, centroids, inertia = kmeans_single_run(
            features_train,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=rstate
        )
        if (best_inertia is None) or (inertia < best_inertia):
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    print("Hoan thanh K-Means tren TAP TRAIN")
    return best_centroids, best_labels

def assign_clusters(features, centroids):
    """
    Gan cluster cho features bat ky, dung centroids da train
    """
    diff = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    return labels

# ==================== 4. MAP CLUSTER SANG LABEL ====================
def map_clusters_to_emotions(cluster_labels_train, true_labels_train):
    """
    Dung TAP TRAIN de map cluster -> emotion pho bien
    """
    print("\nMap cluster tren TAP TRAIN sang emotion pho bien nhat...")
    cluster_map = {}
    for cluster_id in range(N_CLUSTERS):
        mask = (cluster_labels_train == cluster_id)
        emotions_in_cluster = true_labels_train[mask]
        if len(emotions_in_cluster) > 0:
            most_common = Counter(emotions_in_cluster).most_common(1)[0][0]
            cluster_map[cluster_id] = most_common
            print(f"  Cluster {cluster_id} -> {EMOTIONS[most_common]}")
        else:
            print(f"  Cluster {cluster_id} rong")
    mapped_train = np.array([cluster_map.get(c, -1) for c in cluster_labels_train])
    return cluster_map, mapped_train

# ==================== 5. DANH GIA ====================
def evaluate(features, cluster_labels, true_labels, mapped_labels, set_name="TEST"):
    print(f"\n========== DANH GIA TREN TAP {set_name} ==========")
    try:
        silhouette = silhouette_score(features, cluster_labels)
        print(f"Silhouette Score ({set_name}): {silhouette:.4f}")
    except Exception as e:
        silhouette = float('nan')
        print(f"Khong tinh duoc Silhouette ({set_name}):", e)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    acc = accuracy_score(true_labels, mapped_labels)

    print(f"ARI ({set_name}): {ari:.4f}")
    print(f"NMI ({set_name}): {nmi:.4f}")
    print(f"Accuracy sau mapping ({set_name}): {acc:.4f}")

    cm = confusion_matrix(true_labels, mapped_labels)
    print(f"Confusion matrix ({set_name}):")
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
    print("\n==============================================")
    print("FER2013 K-MEANS CLUSTERING - TRAIN/TEST VERSION")
    print("==============================================")

    os.makedirs(FIG_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, TRAIN_DIR_NAME)
    test_path = os.path.join(DATA_DIR, TEST_DIR_NAME)

    # Load tap TRAIN
    X_train_imgs, y_train = load_data(train_path, max_images_per_emotion=300)
    if X_train_imgs.size == 0:
        print("Khong co du lieu TRAIN, thoat")
        return

    # Load tap TEST
    X_test_imgs, y_test = load_data(test_path, max_images_per_emotion=300)
    if X_test_imgs.size == 0:
        print("Khong co du lieu TEST, thoat")
        return

    print(f"\nSo anh TRAIN: {len(X_train_imgs)}")
    print(f"So anh TEST:  {len(X_test_imgs)}")

    # HOG cho TRAIN va TEST
    print("\n=== Trich HOG TAP TRAIN ===")
    X_train_hog = extract_hog_features(X_train_imgs)

    print("\n=== Trich HOG TAP TEST ===")
    X_test_hog = extract_hog_features(X_test_imgs)

    # StandardScaler fit tren TRAIN, transform cho ca TRAIN va TEST
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hog)
    X_test_scaled = scaler.transform(X_test_hog)
    print("\nDa chuan hoa feature bang StandardScaler (fit tren TRAIN)")

    all_results_test = []  # ket qua tren TAP TEST cho tung ty le PCA

    # Thu nghiem nhieu ti le PCA khac nhau
    for ratio in PCA_RATIOS:
        print("\n==============================================")
        print(f"Ap dung PCA, ti le phuong sai yeu cau = {ratio:.2f}")
        print("==============================================")

        # PCA fit tren TAP TRAIN, sau do transform ca TRAIN va TEST
        pca = PCA(n_components=ratio, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        real_ratio = pca.explained_variance_ratio_.sum()
        print(f"Output dim sau PCA (TRAIN): {X_train_pca.shape[1]}")
        print(f"Tong ti le phuong sai thuc te: {real_ratio:.4f}")

        if X_train_pca.shape[0] < N_CLUSTERS:
            print("So mau TRAIN < so cluster, khong chay duoc KMeans o ti le nay")
            continue

        # KMeans tren TAP TRAIN
        centroids, cluster_labels_train = apply_kmeans_train(X_train_pca)

        # Gan cluster cho TAP TEST
        cluster_labels_test = assign_clusters(X_test_pca, centroids)

        # Map cluster -> emotion dua tren TAP TRAIN
        cluster_map, mapped_train = map_clusters_to_emotions(cluster_labels_train, y_train)

        # Map label cho TAP TEST
        mapped_test = np.array([cluster_map.get(c, -1) for c in cluster_labels_test])

        # Danh gia tren TAP TEST
        metrics_test = evaluate(X_test_pca, cluster_labels_test, y_test, mapped_test, set_name="TEST")

        # Ve confusion matrix tren TAP TEST
        cm_test = metrics_test["confusion_matrix"]
        cm_title = f"Confusion Matrix TEST - PCA {int(ratio * 100)}% (var={real_ratio:.2f})"
        cm_path = os.path.join(FIG_DIR, f"cm_test_pca_{int(ratio * 100)}.png")
        plot_confusion_matrix(cm_test, EMOTIONS, title=cm_title, save_path=cm_path)

        all_results_test.append({
            "pca_ratio": ratio,
            "real_ratio": real_ratio,
            "metrics_test": metrics_test,
            "cm_path_test": cm_path
        })

    if not all_results_test:
        print("Khong co ket qua nao tu cac moc PCA, thoat")
        return

    # Tong ket accuracy tren TAP TEST theo tung moc PCA
    print("\n=========== TONG KET ACCURACY TREN TAP TEST THEO PCA ===========")
    for res in all_results_test:
        r = res["pca_ratio"]
        acc = res["metrics_test"]["accuracy"]
        ari = res["metrics_test"]["ari"]
        nmi = res["metrics_test"]["nmi"]
        print(f"PCA yeu cau {r:.2f} -> Accuracy TEST: {acc:.4f}, ARI TEST: {ari:.4f}, NMI TEST: {nmi:.4f}")

    # Tim moc PCA co accuracy TAP TEST cao nhat
    accuracies_test = [res["metrics_test"]["accuracy"] for res in all_results_test]
    best_idx = int(np.argmax(accuracies_test))
    best_res = all_results_test[best_idx]

    print("\n=========== CAU HINH TOT NHAT TREN TAP TEST ===========")
    print(f"PCA yeu cau: {best_res['pca_ratio']:.2f}")
    print(f"Ti le phuong sai thuc te: {best_res['real_ratio']:.4f}")
    print(f"Accuracy TEST: {best_res['metrics_test']['accuracy']:.4f}")
    print(f"ARI TEST:      {best_res['metrics_test']['ari']:.4f}")
    print(f"NMI TEST:      {best_res['metrics_test']['nmi']:.4f}")
    print("Confusion matrix TEST cua cau hinh tot nhat:")
    print(best_res["metrics_test"]["confusion_matrix"])
    print(f"File anh confusion matrix TEST tot nhat: {best_res['cm_path_test']}")
    print("=======================================================")

if __name__ == "__main__":
    main()
