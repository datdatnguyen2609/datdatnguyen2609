"""
Nguyen Thanh Dat - 211440519 - BTL
FER2013 HOG + PCA + K-Means
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

#1 Cau hinh
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

#2 Ve confusion matrix
def plot_confusion_matrix(cm, emotions, title="", save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(emotions))
    plt.xticks(tick_marks, emotions, rotation=45)
    plt.yticks(tick_marks, emotions)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.close()

#3 Load du lieu
def load_data(data_folder, max_images_per_emotion=500):
    images = []
    labels = []

    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(data_folder, emotion)
        if not os.path.exists(emotion_path):
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

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

#4 HOG
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
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size

    hist_cells = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    bin_size = 180.0 / nbins

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size
            cell_mag = mag[y0:y0+cell_size, x0:x0+cell_size].reshape(-1)
            cell_ang = angle[y0:y0+cell_size, x0:x0+cell_size].reshape(-1)

            for m, a in zip(cell_mag, cell_ang):
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

def extract_hog_features(images_flat):
    feats = []
    for img_flat in images_flat:
        img = img_flat.reshape(IMG_SIZE)
        feats.append(hog_single_image(img))
    return np.array(feats)

#5 K-means
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

    return labels, centroids

def apply_kmeans_train(X_train, n_clusters=N_CLUSTERS, max_iter=300, n_init=5, base_random_state=42):
    best_inertia = None
    best_labels = None
    best_centroids = None

    for i in range(n_init):
        labels, centroids = kmeans_single_run(
            X_train,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=base_random_state + i
        )

        diff = X_train[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        inertia = np.sum(np.min(dist_sq, axis=1))

        if (best_inertia is None) or (inertia < best_inertia):
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_centroids, best_labels

def assign_clusters(X, centroids):
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.argmin(dist_sq, axis=1)

#6 Map cluster sang emotion
def map_clusters_to_emotions(cluster_labels_train, y_train):
    cluster_map = {}
    for cluster_id in range(N_CLUSTERS):
        mask = (cluster_labels_train == cluster_id)
        values = y_train[mask]
        if len(values) > 0:
            mc = Counter(values).most_common(1)[0][0]
            cluster_map[cluster_id] = mc
        else:
            cluster_map[cluster_id] = -1
    return cluster_map

def convert_cluster_to_labels(cluster_labels, cluster_map):
    return np.array([cluster_map.get(c, -1) for c in cluster_labels])

#7 Danh gia
def evaluate_on_test(y_test, y_test_pred):
    acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    return acc, cm

#8 Main
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, TRAIN_DIR_NAME)
    test_path = os.path.join(DATA_DIR, TEST_DIR_NAME)

    X_train_imgs, y_train = load_data(train_path, max_images_per_emotion=300)
    X_test_imgs, y_test = load_data(test_path, max_images_per_emotion=300)

    X_train_hog = extract_hog_features(X_train_imgs)
    X_test_hog = extract_hog_features(X_test_imgs)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hog)
    X_test_scaled = scaler.transform(X_test_hog)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    real_ratio = pca.explained_variance_ratio_.sum()

    centroids, cluster_labels_train = apply_kmeans_train(X_train_pca)

    cluster_map = map_clusters_to_emotions(cluster_labels_train, y_train)

    cluster_labels_test = assign_clusters(X_test_pca, centroids)
    y_test_pred = convert_cluster_to_labels(cluster_labels_test, cluster_map)

    acc_test, cm_test = evaluate_on_test(y_test, y_test_pred)

    title = f"K-Means Confusion Matrix TEST PCA {int(PCA_COMPONENTS * 100)} (var={real_ratio:.2f})"
    cm_path = os.path.join(FIG_DIR, f"kmeans_cm_test_pca_{int(PCA_COMPONENTS * 100)}.png")
    plot_confusion_matrix(cm_test, EMOTIONS, title=title, save_path=cm_path)

    print("PCA:", PCA_COMPONENTS)
    print("Variance:", real_ratio)
    print("Accuracy TEST:", acc_test)
    print("Saved CM:", cm_path)

if __name__ == "__main__":
    main()
