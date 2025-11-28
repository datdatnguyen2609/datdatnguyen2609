"""
FER2013 Clustering - HOG + PCA + GMM Version
Nang cap: dung HOG + StandardScaler + PCA + GaussianMixture de cai thien chat luong phan cum
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from collections import Counter

import seaborn as sns
from skimage.feature import hog

# ==================== CAU HINH ====================
DATA_DIR = "fer2013"  # Thu muc chua du lieu
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
N_CLUSTERS = 7
IMG_SIZE = (48, 48)

USE_HOG = True            # Dung HOG lam dac trung
USE_PCA = True            # Dung PCA sau HOG
PCA_COMPONENTS = 0.95     # 0 < x < 1: ti le phuong sai, o day la 95 phan tram

# Cau hinh HOG
HOG_CELL_SIZE = 6         # kich thuoc cell 6x6
HOG_BLOCK_SIZE = 2        # so cell moi block 2x2
HOG_BINS = 9              # so bin huong

# ==================== 1. LOAD DU LIEU ====================
def load_data(data_folder, max_images_per_emotion=500):
    """
    Load anh tu thu muc FER2013
    data_folder: duong dan thu muc, vi du "fer2013/training"
    max_images_per_emotion: gioi han so anh moi class de test nhanh
    Tra ve:
        images: ndarray (N, 48*48), da normalize ve [0, 1]
        labels: ndarray (N,) la nhan 0..6 tuong ung EMOTIONS
    """
    print(f"\nDang load du lieu tu {data_folder}...")

    images = []
    labels = []

    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(data_folder, emotion)

        if not os.path.exists(emotion_path):
            print(f"Khong tim thay thu muc: {emotion_path}")
            continue

        files = os.listdir(emotion_path)[:max_images_per_emotion]

        for img_file in files:
            img_path = os.path.join(emotion_path, img_file)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)
                img = img.astype("float32") / 255.0

                # Loai bo anh rat xau, gan nhu toan mau xam
                if img.std() < 0.02:
                    continue

                images.append(img.flatten())
                labels.append(idx)

            except Exception:
                continue

        count_emotion = len([l for l in labels if l == idx])
        print(f"  {emotion}: {count_emotion} anh")

    images = np.array(images)
    labels = np.array(labels)

    print(f"\nTong so anh load duoc: {images.shape[0]}")
    print(f"Input flatten shape: {images.shape}")

    if images.size == 0:
        print("Khong co anh nao duoc load. Kiem tra lai DATA_DIR va cau truc thu muc.")

    return images, labels

# ==================== 1b. HOG FEATURE EXTRACTION ====================
def extract_hog_features(images_flat, img_size=IMG_SIZE):
    """
    Trich dac trung HOG
    images_flat: ndarray (N, 48*48), da normalize
    Tra ve:
        hog_features: ndarray (N, D_hog)
    """
    print("\nTrich dac trung HOG...")
    n_samples = images_flat.shape[0]
    hog_features = []

    for i in range(n_samples):
        img = images_flat[i].reshape(img_size)

        feat = hog(
            img,
            orientations=HOG_BINS,
            pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
            cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True
        )
        hog_features.append(feat)

        if (i + 1) % 500 == 0:
            print(f"  Da trich HOG {i + 1} anh...")

    hog_features = np.array(hog_features)
    print(f"  HOG feature shape: {hog_features.shape}")
    return hog_features

# ==================== 2. GMM CLUSTERING ====================
def apply_gmm(features, n_clusters=N_CLUSTERS):
    """
    Ap dung Gaussian Mixture Model clustering
    features: ndarray (N, D), nen la float64
    Tra ve:
        gmm: model da fit
        cluster_labels: ndarray (N,)
    """
    print(f"\nDang chay Gaussian Mixture voi {n_clusters} clusters...")

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",   # dung ma tran hiep phuong sai duong cheo cho on dinh hon
        random_state=42,
        n_init=10,                # chay nhieu lan hon
        max_iter=500,
        reg_covar=1e-3            # regularization de tranh covariance suy bien
    )

    cluster_labels = gmm.fit_predict(features)
    print("  Hoan thanh Gaussian Mixture")
    return gmm, cluster_labels


# ==================== 3. MAP CLUSTERS -> LABELS ====================
def map_clusters_to_emotions(cluster_labels, true_labels):
    """
    Map moi cluster voi emotion pho bien nhat
    cluster_labels: nhan cluster cua GMM
    true_labels: nhan ground truth
    Tra ve:
        cluster_map: dict cluster_id -> emotion_id
        mapped_labels: ndarray mapped tu cluster sang label
    """
    print("\nDang map clusters voi emotions...")

    cluster_map = {}

    for cluster_id in range(N_CLUSTERS):
        mask = cluster_labels == cluster_id
        emotions_in_cluster = true_labels[mask]

        if len(emotions_in_cluster) > 0:
            most_common = Counter(emotions_in_cluster).most_common(1)[0][0]
            cluster_map[cluster_id] = most_common
            print(f"  Cluster {cluster_id} -> {EMOTIONS[most_common]}")
        else:
            print(f"  Cluster {cluster_id} khong co mau nao, bo qua")

    mapped_labels = np.array([cluster_map.get(c, -1) for c in cluster_labels])

    return cluster_map, mapped_labels

# ==================== 4. DANH GIA ====================
def evaluate(features, cluster_labels, true_labels, mapped_labels):
    """
    Danh gia ket qua clustering
    Tra ve dict metrics
    """
    print("\n" + "=" * 60)
    print("KET QUA DANH GIA")
    print("=" * 60)

    try:
        silhouette = silhouette_score(features, cluster_labels)
        print(f"\nSilhouette Score: {silhouette:.4f}  cao hon la tot hon")
    except Exception as e:
        silhouette = float("nan")
        print("\nKhong tinh duoc Silhouette Score do loi:", e)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    print(f"ARI: {ari:.4f}  range: -1 den 1, cao hon la tot hon")
    print(f"NMI: {nmi:.4f}  range: 0 den 1, cao hon la tot hon")

    accuracy = accuracy_score(true_labels, mapped_labels)
    print(f"Accuracy sau mapping: {accuracy:.4f}")

    cm = confusion_matrix(true_labels, mapped_labels)

    return {
        "silhouette": silhouette,
        "ari": ari,
        "nmi": nmi,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

# ==================== 5. VISUALIZATION ====================
def visualize_results(features, cluster_labels, true_labels, confusion_mat):
    """
    Ve bieu do ket qua
    Luu:
        outputs/confusion_matrix.png
        outputs/tsne_comparison.png
    """
    print("\nDang tao cac bieu do visualization...")

    os.makedirs("outputs", exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS
    )
    plt.title("Confusion Matrix: GMM vs Ground Truth", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted   from GMM")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    print("  Saved: outputs/confusion_matrix.png")
    plt.close()

    # 2. t-SNE Visualization
    print("  Dang tinh t-SNE  neu du lieu lon co the hoi cham...")

    n_samples = min(2000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features[indices])

    plt.figure(figsize=(12, 5))

    # Plot clusters cua GMM
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=cluster_labels[indices],
        cmap="tab10",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title("t-SNE: GMM Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Plot true labels
    plt.subplot(1, 2, 2)
    for idx, emotion in enumerate(EMOTIONS):
        mask = true_labels[indices] == idx
        plt.scatter(
            tsne_features[mask, 0],
            tsne_features[mask, 1],
            label=emotion,
            alpha=0.6,
            edgecolors="w",
            linewidth=0.5
        )
    plt.title("t-SNE: True Emotions")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/tsne_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved: outputs/tsne_comparison.png")
    plt.close()

# ==================== MAIN FUNCTION ====================
def main():
    print("\n" + "=" * 60)
    print("FER2013 CLUSTERING - HOG + PCA + GMM VERSION")
    print("=" * 60)

    # 1  Load du lieu
    train_path = os.path.join(DATA_DIR, "training")

    train_images, train_labels = load_data(train_path, max_images_per_emotion=300)

    images = train_images
    labels = train_labels

    if images.size == 0:
        print("\nKhong co du lieu de xu ly. Ket thuc chuong trinh.")
        return

    print(f"\nTong so anh: {len(images)}")
    print(f"Original feature shape  flatten pixel: {images.shape}")

    # 2  Trich HOG
    features = images
    if USE_HOG:
        features = extract_hog_features(images)
        print(f"Feature shape sau HOG: {features.shape}")

    # 2b  Chuan hoa feature
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Da chuan hoa feature bang StandardScaler.")

    if features.ndim == 1:
        features = features.reshape(-1, 1)
        print(f"Feature duoc reshape thanh: {features.shape}")

    if features.shape[0] < 2:
        print("\nCan it nhat 2 mau de chay PCA va GMM. Hien tai chi co "
              f"{features.shape[0]} mau. Ket thuc chuong trinh.")
        return

    # 3  PCA  optional
    if USE_PCA:
        print(f"\nAp dung PCA, input dim = {features.shape[1]}")
        print(f"n_components = {PCA_COMPONENTS}  neu 0 < x < 1 la ti le phuong sai")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        features = pca.fit_transform(features)
        print(f"  Output dim sau PCA: {features.shape[1]}")
        print(f"  Tong ti le phuong sai giai thich: {pca.explained_variance_ratio_.sum():.4f}")

    print(f"\nFeature final shape dua vao GMM: {features.shape}")

    if features.shape[0] < N_CLUSTERS:
        print(f"\nCan it nhat {N_CLUSTERS} mau de chia {N_CLUSTERS} cum "
              f"nhung hien tai chi co {features.shape[0]} mau. Ket thuc chuong trinh.")
        return

    # 4  GMM Clustering
    print(f"\nFeature final shape dua vao GMM: {features.shape}")

    if features.shape[0] < N_CLUSTERS:
        print(f"\nCan it nhat {N_CLUSTERS} mau de chia {N_CLUSTERS} cum "
              f"nhung hien tai chi co {features.shape[0]} mau. Ket thuc chuong trinh.")
        return

    # Ep kieu ve float64 cho GMM de so
    features = features.astype(np.float64)

    # 4  GMM Clustering
    gmm, cluster_labels = apply_gmm(features)

    # 5  Map clusters -> emotions
    cluster_map, mapped_labels = map_clusters_to_emotions(cluster_labels, labels)

    # 6  Danh gia
    metrics = evaluate(features, cluster_labels, labels, mapped_labels)

    # 7  Visualization
    visualize_results(features, cluster_labels, labels, metrics["confusion_matrix"])

    # 8  Ket luan
    print("\n" + "=" * 60)
    print("HOAN THANH")
    print("=" * 60)
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print("\nKet qua da duoc luu vao thu muc outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
