"""
Nguyễn Thành Đạt - 211440519 - BTL
FER2013 K-Means Clustering - HOG + PCA Version
Nâng cấp: dùng HOG + PCA + StandardScaler, nhưng:
- HOG tự cài đặt, không dùng thư viện hog
- K-Means tự cài đặt, không dùng sklearn.cluster.KMeans
- PCA vẫn dùng thư viện
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
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

# ==================== CẤU HÌNH ====================
DATA_DIR = "fer2013"  # Thư mục chứa dữ liệu
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
N_CLUSTERS = 7
IMG_SIZE = (48, 48)

USE_HOG = True           # Dùng HOG làm đặc trưng
USE_PCA = True           # Dùng PCA sau HOG
PCA_COMPONENTS = 0.95    # 0 < x < 1: tỉ lệ phương sai, ở đây là 95%

# Cấu hình HOG
HOG_CELL_SIZE = 6        # kích thước cell 6x6
HOG_BLOCK_SIZE = 2       # số cell mỗi block 2x2
HOG_BINS = 9             # số bin hướng

# ==================== 1. LOAD DỮ LIỆU ====================
def load_data(data_folder, max_images_per_emotion=500):
    """Load ảnh từ thư mục"""
    print(f"\nĐang load dữ liệu từ {data_folder}...")
    
    images = []
    labels = []
    
    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(data_folder, emotion)
        
        if not os.path.exists(emotion_path):
            print(f"Không tìm thấy thư mục: {emotion_path}")
            continue
        
        files = os.listdir(emotion_path)[:max_images_per_emotion]
        
        for img_file in files:
            img_path = os.path.join(emotion_path, img_file)
            
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)
                img = img.astype('float32') / 255.0  # Normalize về [0, 1]

                # Lọc bớt ảnh rất xấu
                if img.std() < 0.02:
                    continue
                
                images.append(img.flatten())
                labels.append(idx)
                
            except Exception:
                continue
        
        count_emotion = len([l for l in labels if l == idx])
        print(f"  {emotion}: {count_emotion} ảnh")
    
    images = np.array(images)
    labels = np.array(labels)

    print(f"\nTổng số ảnh load được: {images.shape[0]}")
    print(f"Input flatten shape: {images.shape}")
    
    if images.size == 0:
        print("Không có ảnh nào được load. Kiểm tra lại DATA_DIR và cấu trúc thư mục.")
    
    return images, labels

# ==================== 1b. HOG TỰ CÀI ĐẶT ====================
def hog_single_image(img,
                     cell_size=HOG_CELL_SIZE,
                     block_size=HOG_BLOCK_SIZE,
                     nbins=HOG_BINS):
    """
    Tính HOG cho 1 ảnh (48x48, float32, [0,1])
    Không dùng skimage.feature.hog
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Tính gradient gx, gy bằng Sobel
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Độ lớn và góc gradient
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Dùng gradient không dấu trong [0, 180)
    angle = angle % 180.0

    h, w = img.shape
    cell_h = cell_w = cell_size
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    # Khởi tạo histogram cho từng cell
    hist_cells = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)

    bin_size = 180.0 / nbins

    # Tính histogram cho từng cell (không nội suy, gán cứng vào bin gần nhất)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y_start = cy * cell_h
            y_end = y_start + cell_h
            x_start = cx * cell_w
            x_end = x_start + cell_w

            cell_mag = mag[y_start:y_end, x_start:x_end]
            cell_angle = angle[y_start:y_end, x_start:x_end]

            cell_mag_flat = cell_mag.reshape(-1)
            cell_angle_flat = cell_angle.reshape(-1)

            for m, a in zip(cell_mag_flat, cell_angle_flat):
                bin_idx = int(a // bin_size)
                if bin_idx == nbins:
                    bin_idx = nbins - 1
                hist_cells[cy, cx, bin_idx] += m

    # Chuẩn hóa block 2x2 cell
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

    hog_vector = np.array(hog_vector, dtype=np.float32)
    return hog_vector

def extract_hog_features(images_flat, img_size=IMG_SIZE):
    """
    images_flat: array shape (N, 48*48) đã normalize
    Trả về: HOG features shape (N, D_hog)
    """
    print("\nTrích đặc trưng HOG (tự cài đặt)...")
    n_samples = images_flat.shape[0]
    hog_features = []

    for i in range(n_samples):
        img = images_flat[i].reshape(img_size)
        feat = hog_single_image(img)
        hog_features.append(feat)

        if (i + 1) % 500 == 0:
            print(f"  Đã trích HOG {i + 1} ảnh...")

    hog_features = np.array(hog_features)
    print(f"  HOG feature shape: {hog_features.shape}")
    return hog_features

# ==================== 2. K-MEANS TỰ CÀI ĐẶT ====================
def kmeans_single_run(X, n_clusters, max_iter=300, random_state=None):
    """
    Chạy 1 lần K-Means đơn giản
    X: (N, D)
    Trả về: labels, centroids, inertia
    """
    N, D = X.shape
    rng = np.random.RandomState(random_state)

    # Chọn ngẫu nhiên tâm ban đầu từ dữ liệu
    init_idx = rng.choice(N, n_clusters, replace=False)
    centroids = X[init_idx].copy()

    for it in range(max_iter):
        # Tính khoảng cách bình phương tới các centroid
        # dist shape: (N, K)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)

        labels = np.argmin(dist_sq, axis=1)

        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # Nếu cluster rỗng, khởi tạo lại ngẫu nhiên
                new_centroids[k] = X[rng.randint(0, N)]

        shift = np.sqrt(np.sum((centroids - new_centroids) ** 2))
        centroids = new_centroids

        if shift < 1e-4:
            break

    # Tính inertia
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    closest = np.min(dist_sq, axis=1)
    inertia = np.sum(closest)

    return labels, centroids, inertia

def apply_kmeans(features, n_clusters=N_CLUSTERS, n_init=10, max_iter=500, base_random_state=42):
    """Áp dụng K-Means tự cài đặt, chạy nhiều lần lấy kết quả tốt nhất"""
    print(f"\nĐang chạy K-Means tự cài đặt với {n_clusters} clusters...")

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

    print("  Hoàn thành K-Means")
    return best_centroids, best_labels

# ==================== 3. MAP CLUSTERS → LABELS ====================
def map_clusters_to_emotions(cluster_labels, true_labels):
    """Map mỗi cluster với emotion phổ biến nhất"""
    print("\nĐang map clusters với emotions...")
    
    cluster_map = {}
    
    for cluster_id in range(N_CLUSTERS):
        mask = cluster_labels == cluster_id
        emotions_in_cluster = true_labels[mask]
        
        if len(emotions_in_cluster) > 0:
            most_common = Counter(emotions_in_cluster).most_common(1)[0][0]
            cluster_map[cluster_id] = most_common
            print(f"  Cluster {cluster_id} -> {EMOTIONS[most_common]}")
        else:
            print(f"  Cluster {cluster_id} không có mẫu nào, bỏ qua")
    
    mapped_labels = np.array([cluster_map.get(c, -1) for c in cluster_labels])
    
    return cluster_map, mapped_labels

# ==================== 4. ĐÁNH GIÁ ====================
def evaluate(features, cluster_labels, true_labels, mapped_labels):
    """Đánh giá kết quả"""
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*60)
    
    # Unsupervised metrics
    try:
        silhouette = silhouette_score(features, cluster_labels)
        print(f"\nSilhouette Score: {silhouette:.4f}  cao hơn là tốt hơn")
    except Exception as e:
        silhouette = float('nan')
        print("\nKhông tính được Silhouette Score do lỗi:", e)
    
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    print(f"ARI: {ari:.4f}  range: -1 đến 1, cao hơn là tốt hơn")
    print(f"NMI: {nmi:.4f}  range: 0 đến 1, cao hơn là tốt hơn")
    
    accuracy = accuracy_score(true_labels, mapped_labels)
    print(f"Accuracy sau mapping: {accuracy:.4f}")
    
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
    """Vẽ biểu đồ kết quả"""
    print("\nĐang tạo các biểu đồ visualization...")
    
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
    print("  Đang tính t-SNE  dữ liệu lớn có thể hơi chậm...")
    
    n_samples = min(2000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features[indices])
    
    plt.figure(figsize=(12, 5))
    
    # t-SNE clusters
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
    
    # t-SNE true labels
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
    """Hàm chính"""
    print("\n" + "="*60)
    print("FER2013 K-MEANS CLUSTERING - HOG TỰ CÀI ĐẶT + PCA VERSION")
    print("="*60)
    
    # 1  Load dữ liệu
    train_path = os.path.join(DATA_DIR, "training")
    
    train_images, train_labels = load_data(train_path, max_images_per_emotion=300)
    
    images = train_images
    labels = train_labels
    
    if images.size == 0:
        print("\nKhông có dữ liệu để xử lý. Kết thúc chương trình.")
        return
    
    print(f"\nTổng số ảnh: {len(images)}")
    print(f"Original feature shape  flatten pixel: {images.shape}")
    
    # 2  HOG features
    features = images
    if USE_HOG:
        features = extract_hog_features(images)
        print(f"Feature shape sau HOG: {features.shape}")
    
    # 2b  Chuẩn hóa feature sau HOG
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print("Đã chuẩn hóa feature bằng StandardScaler.")
    
    if features.ndim == 1:
        features = features.reshape(-1, 1)
        print(f"Feature được reshape thành: {features.shape}")
    
    if features.shape[0] < 2:
        print("\nCần ít nhất 2 mẫu để chạy PCA và KMeans. Hiện tại chỉ có "
              f"{features.shape[0]} mẫu. Kết thúc chương trình.")
        return
    
    # 3  PCA  optional
    if USE_PCA:
        print(f"\nÁp dụng PCA, input dim = {features.shape[1]}")
        print(f"n_components = {PCA_COMPONENTS}  nếu 0 < x < 1 là tỉ lệ phương sai")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        features = pca.fit_transform(features)
        print(f"  Output dim sau PCA: {features.shape[1]}")
        print(f"  Tổng tỉ lệ phương sai giải thích: {pca.explained_variance_ratio_.sum():.4f}")
    
    print(f"\nFeature final shape đưa vào KMeans: {features.shape}")
    
    if features.shape[0] < N_CLUSTERS:
        print(f"\nCần ít nhất {N_CLUSTERS} mẫu để chia {N_CLUSTERS} cụm "
              f"nhưng hiện tại chỉ có {features.shape[0]} mẫu. Kết thúc chương trình.")
        return
    
    # 4  KMeans Clustering tự cài đặt
    centroids, cluster_labels = apply_kmeans(features)
    
    # 5  Map clusters to emotions
    cluster_map, mapped_labels = map_clusters_to_emotions(cluster_labels, labels)
    
    # 6  Đánh giá
    metrics = evaluate(features, cluster_labels, labels, mapped_labels)
    
    # 7  Visualization
    visualize_results(features, cluster_labels, labels, metrics['confusion_matrix'])
    
    # 8  Kết luận
    print("\n" + "="*60)
    print("HOÀN THÀNH")
    print("="*60)
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print("\nKết quả đã được lưu vào thư mục 'outputs/'")
    print("="*60)

if __name__ == "__main__":
    main()
