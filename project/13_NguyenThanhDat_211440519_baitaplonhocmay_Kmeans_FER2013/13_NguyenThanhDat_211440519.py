"""
Nguyen Thanh Dat - 211440519 - BTL
FER2013 HOG + PCA + K-Means (train/test version)
- HOG tu cai dat
- PCA dung thu vien (PCA = 0.90)
- K-Means tu cai dat
- Train tren tap training, danh gia tren tap test
- Ve confusion matrix tren TAP TEST va luu file
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ==================== CAU HINH ====================
DATA_DIR = "fer2013"
TRAIN_DIR_NAME = "training"
TEST_DIR_NAME = "test"

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
N_CLASSES = len(EMOTIONS)
N_CLUSTERS = 7

IMG_SIZE = (48, 48)

HOG_CELL_SIZE = 6
HOG_BLOCK_SIZE = 2
HOG_BINS = 9

PCA_COMPONENTS = 0.90

FIG_DIR = "figs_kmeans"

# ==================== 0. HAM VE CONFUSION MATRIX ====================
def plot_confusion_matrix(cm, emotions, title="", save_path=None):
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
                # loai bo anh qua toi / qua sang
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

    # gradient theo x va y
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # do lon va goc gradient
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180.0

    h, w = img.shape
    cell_h = cell_w = cell_size
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    hist_cells = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    bin_size = 180.0 / nbins

    # tinh histogram HOG cho tung cell
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

    # chuan hoa theo block
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

    # khoi tao tam cum ngau nhien
    init_idx = rng.choice(N, n_clusters, replace=False)
    centroids = X[init_idx].copy()

    for _ in range(max_iter):
        # gan cum
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)

        # cap nhat tam cum
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # neu cum rong thi chon ngau nhien lai
                new_centroids[k] = X[rng.randint(0, N)]

        # kiem tra hoi tu
        shift = np.sqrt(np.sum((centroids - new_centroids) ** 2))
        centroids = new_centroids
        if shift < 1e-4:
            break

    return labels, centroids

def apply_kmeans_train(X_train,
                       n_clusters=N_CLUSTERS,
                       max_iter=300,
                       n_init=5,
                       base_random_state=42):
    print(f"\nChay K-Means tu cai dat tren TAP TRAIN, K = {n_clusters}...")
    best_inertia = None
    best_labels = None
    best_centroids = None

    for i in range(n_init):
        rstate = base_random_state + i
        labels, centroids = kmeans_single_run(
            X_train,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=rstate
        )

        # tinh inertia cho lan chay nay
        diff = X_train[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        closest = np.min(dist_sq, axis=1)
        inertia = np.sum(closest)

        if (best_inertia is None) or (inertia < best_inertia):
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    print("Hoan thanh K-Means tren TAP TRAIN")
    return best_centroids, best_labels

def assign_clusters(X, centroids):
    """
    Gan cum cho tap du lieu moi (VD TAP TEST) dua tren cac centroids da hoc duoc tu TRAIN
    """
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    return labels

# ==================== 4. MAP CLUSTER SANG EMOTION ====================
def map_clusters_to_emotions(cluster_labels_train, y_train):
    """
    Map moi cluster sang emotion pho bien nhat tren TAP TRAIN
    Sau do dung mapping nay de du doan label cho TAP TEST
    """
    print("\nMap cluster sang emotion pho bien nhat tren TAP TRAIN...")
    cluster_map = {}
    for cluster_id in range(N_CLUSTERS):
        mask = (cluster_labels_train == cluster_id)
        emotions_in_cluster = y_train[mask]
        if len(emotions_in_cluster) > 0:
            most_common = Counter(emotions_in_cluster).most_common(1)[0][0]
            cluster_map[cluster_id] = most_common
            print(f"  Cluster {cluster_id} -> {EMOTIONS[most_common]}")
        else:
            print(f"  Cluster {cluster_id} rong (khong co mau train)")
    return cluster_map

def convert_cluster_to_labels(cluster_labels, cluster_map):
    mapped_labels = np.array([cluster_map.get(c, -1) for c in cluster_labels])
    return mapped_labels

# ==================== 5. DANH GIA TREN TAP TEST ====================
def evaluate_on_test(y_test, y_test_pred):
    print("\n========== DANH GIA TREN TAP TEST (K-MEANS) ==========")
    acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"Accuracy TEST: {acc:.4f}")
    print("Confusion matrix TEST:")
    print(cm)

    return acc, cm

# ==================== MAIN ====================
def main():
    print("\n==================================================")
    print("FER2013 HOG + PCA + K-MEANS - TRAIN/TEST VERSION")
    print("==================================================")

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

    # StandardScaler fit tren TRAIN, transform ca TRAIN va TEST
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hog)
    X_test_scaled = scaler.transform(X_test_hog)
    print("\nDa chuan hoa feature bang StandardScaler (fit tren TRAIN)")

    # PCA fit tren TRAIN, transform ca TRAIN va TEST
    print(f"\nAp dung PCA, ti le phuong sai yeu cau = {PCA_COMPONENTS:.2f}")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    real_ratio = pca.explained_variance_ratio_.sum()
    print(f"Output dim sau PCA (TRAIN): {X_train_pca.shape[1]}")
    print(f"Tong ti le phuong sai thuc te: {real_ratio:.4f}")

    # K-Means train tren TAP TRAIN
    centroids, cluster_labels_train = apply_kmeans_train(X_train_pca)

    # Map cluster -> emotion dua tren TAP TRAIN
    cluster_map = map_clusters_to_emotions(cluster_labels_train, y_train)

    # Gan cum cho TAP TEST
    cluster_labels_test = assign_clusters(X_test_pca, centroids)

    # Chuyen cluster cua TAP TEST sang label cam xuc
    y_test_pred = convert_cluster_to_labels(cluster_labels_test, cluster_map)

    # Danh gia tren TAP TEST
    acc_test, cm_test = evaluate_on_test(y_test, y_test_pred)

    # Ve confusion matrix tren TAP TEST
    cm_title = f"K-Means Confusion Matrix TEST - PCA {int(PCA_COMPONENTS * 100)}% (var={real_ratio:.2f})"
    cm_path = os.path.join(FIG_DIR, f"kmeans_cm_test_pca_{int(PCA_COMPONENTS * 100)}.png")
    plot_confusion_matrix(cm_test, EMOTIONS, title=cm_title, save_path=cm_path)

    print("\n=========== TONG KET ===========")
    print(f"PCA yeu cau: {PCA_COMPONENTS:.2f}")
    print(f"Ti le phuong sai thuc te: {real_ratio:.4f}")
    print(f"Accuracy TEST (K-Means): {acc_test:.4f}")
    print(f"File anh confusion matrix TEST: {cm_path}")
    print("================================")

if __name__ == "__main__":
    main()
