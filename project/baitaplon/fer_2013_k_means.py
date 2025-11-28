"""
FER2013 K-Means Clustering - HOG + PCA Version
Nang cap: dung HOG + PCA + StandardScaler de cai thien chat luong phan cum
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
N_CLUSTERS = 7
IMG_SIZE = (48, 48)

USE_HOG = True           # Dung HOG lam dac trung
USE_PCA = True           # Dung PCA sau HOG
PCA_COMPONENTS = 0.95    # 0 < x < 1: ti le phuong sai, o day la 95 phan tram

# Cau hinh HOG (ban co the thu nghiem thay doi)
HOG_CELL_SIZE = 6        # kich thuoc cell 6x6 (chi tiet hon 8x8)
HOG_BLOCK_SIZE = 2       # so cell moi block 2x2
HOG_BINS = 9             # so bin huong

# ==================== 1. LOAD DU LIEU ====================
def load_data(data_folder, max_images_per_emotion=500):
    """Load anh tu thu muc"""
    print(f"\nDang load du lieu tu {data_folder}...")
    
    images = []
    labels = []
    
    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(data_folder, emotion)
        
        if not os.path.exists(emotion_path):
            print(f"Khong tim thay thu muc: {emotion_path}")
            continue
        
        files = os.listdir(emotion_path)[:max_images_per_emotion]  # Gioi han so anh de test nhanh
        
        for img_file in files:
            img_path = os.path.join(emotion_path, img_file)
            
            try:
                # Doc anh grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)
                img = img.astype('float32') / 255.0  # Normalize

                # Loc bot anh rat xau (gan nhu toan mau xam)
                if img.std() < 0.02:
                    continue
                
                images.append(img.flatten())  # Flatten thanh vector 1D
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
    images_flat: array shape (N, 48*48) da normalize
    Tra ve: HOG features shape (N, D_hog)
    """
    print("\nTrich dac trung HOG...")
    n_samples = images_flat.shape[0]
    hog_features = []

    for i in range(n_samples):
        img = images_flat[i].reshape(img_size)  # reshape ve 48x48

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

        # In nho de check tien trinh neu data lon
        if (i + 1) % 500 == 0:
            print(f"  Da trich HOG {i + 1} anh...")

    hog_features = np.array(hog_features)
    print(f"  HOG feature shape: {hog_features.shape}")
    return hog_features

# ==================== 2. K-MEANS CLUSTERING ====================
def apply_kmeans(features, n_clusters=N_CLUSTERS):
    """Ap dung K-Means"""
    print(f"\nDang chay K-Means voi {n_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=30,     # chay nhieu lan de chon ket qua tot hon
        max_iter=500   # cho phep hoi tu ky hon
    )
    
    cluster_labels = kmeans.fit_predict(features)
    print("  Hoan thanh K-Means")
    
    return kmeans, cluster_labels

# ==================== 3. MAP CLUSTERS → LABELS ====================
def map_clusters_to_emotions(cluster_labels, true_labels):
    """Map moi cluster voi emotion pho bien nhat"""
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
    
    # Chuyen cluster predictions thanh emotion predictions
    mapped_labels = np.array([cluster_map.get(c, -1) for c in cluster_labels])
    
    return cluster_map, mapped_labels

# ==================== 4. DANH GIA ====================
def evaluate(features, cluster_labels, true_labels, mapped_labels):
    """Danh gia ket qua"""
    print("\n" + "="*60)
    print("KET QUA DANH GIA")
    print("="*60)
    
    # Unsupervised metrics
    try:
        silhouette = silhouette_score(features, cluster_labels)
        print(f"\nSilhouette Score: {silhouette:.4f}  cao hon la tot hon")
    except Exception as e:
        silhouette = float('nan')
        print("\nKhong tinh duoc Silhouette Score do loi:", e)
    
    # So sanh voi ground truth
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    print(f"ARI: {ari:.4f}  range: -1 den 1, cao hon la tot hon")
    print(f"NMI: {nmi:.4f}  range: 0 den 1, cao hon la tot hon")
    
    # Accuracy sau khi map
    accuracy = accuracy_score(true_labels, mapped_labels)
    print(f"Accuracy  sau mapping: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, mapped_labels)
    
    return {
        'silhouette': silhouette,
        'ari': ari,
        'nmi': nmi,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

# ==================== 5. VISUALIZATION ====================
def visualize_results(features, cluster_labels, true_labels, confusion_mat):
    """Ve bieu do ket qua"""
    print("\nDang tao cac bieu do visualization...")
    
    # Tao output folder
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS
    )
    plt.title('Confusion Matrix: K-Means vs Ground Truth', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted  K-Means')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    print("  Saved: outputs/confusion_matrix.png")
    plt.close()
    
    # 2. t-SNE Visualization
    print("  Dang tinh t-SNE  neu du lieu lon co the hoi cham...")
    
    # Sample data neu qua nhieu  de t-SNE chay nhanh hon
    n_samples = min(2000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features[indices])
    
    # Plot clusters
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=cluster_labels[indices],
        cmap='tab10',
        alpha=0.6,
        edgecolors='w',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE: K-Means Clusters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # Plot true labels
    plt.subplot(1, 2, 2)
    for idx, emotion in enumerate(EMOTIONS):
        mask = true_labels[indices] == idx
        plt.scatter(
            tsne_features[mask, 0],
            tsne_features[mask, 1],
            label=emotion,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
    plt.title('t-SNE: True Emotions')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/tsne_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: outputs/tsne_comparison.png")
    plt.close()

# ==================== MAIN FUNCTION ====================
def main():
    """Ham chinh"""
    print("\n" + "="*60)
    print("FER2013 K-MEANS CLUSTERING - HOG + PCA VERSION")
    print("="*60)
    
    # 1  Load du lieu
    train_path = os.path.join(DATA_DIR, "training")
    
    train_images, train_labels = load_data(train_path, max_images_per_emotion=300)
    
    images = train_images
    labels = train_labels
    
    # Kiem tra co du lieu hay khong
    if images.size == 0:
        print("\nKhong co du lieu de xu ly. Ket thuc chuong trinh.")
        return
    
    print(f"\nTong so anh: {len(images)}")
    print(f"Original feature shape  flatten pixel: {images.shape}")
    
    # 2  HOG features
    features = images
    if USE_HOG:
        features = extract_hog_features(images)
        print(f"Feature shape sau HOG: {features.shape}")
    
    # 2b  Chuan hoa feature sau HOG
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Da chuan hoa feature bang StandardScaler.")
    
    # Dam bao features luon 2 chieu
    if features.ndim == 1:
        features = features.reshape(-1, 1)
        print(f"Feature duoc reshape thanh: {features.shape}")
    
    # Kiem tra so mau toi thieu
    if features.shape[0] < 2:
        print("\nCan it nhat 2 mau de chay PCA va KMeans. Hien tai chi co "
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
    
    print(f"\nFeature final shape dua vao KMeans: {features.shape}")
    
    # Neu so mau nho hon so cum thi KMeans rat yeu
    if features.shape[0] < N_CLUSTERS:
        print(f"\nCan it nhat {N_CLUSTERS} mau de chia {N_CLUSTERS} cum "
              f"nhung hien tai chi co {features.shape[0]} mau. Ket thuc chuong trinh.")
        return
    
    # 4  KMeans Clustering
    kmeans, cluster_labels = apply_kmeans(features)
    
    # 5  Map clusters to emotions
    cluster_map, mapped_labels = map_clusters_to_emotions(cluster_labels, labels)
    
    # 6  Danh gia
    metrics = evaluate(features, cluster_labels, labels, mapped_labels)
    
    # 7  Visualization
    visualize_results(features, cluster_labels, labels, metrics['confusion_matrix'])
    
    # 8  Ket luan
    print("\n" + "="*60)
    print("HOAN THANH")
    print("="*60)
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print("\nKet qua da duoc luu vao thu muc 'outputs/'")
    print("="*60)

if __name__ == "__main__":
    main()
