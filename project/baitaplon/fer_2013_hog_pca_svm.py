"""
FER2013 Emotion Recognition - HOG + PCA + LinearSVC
Su dung cung pipeline HOG + PCA nhung training co giam sat de tang accuracy
"""

import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "fer2013"
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (48, 48)

HOG_CELL_SIZE = 6
HOG_BLOCK_SIZE = 2
HOG_BINS = 9

PCA_COMPONENTS = 0.95    # giu 95 phan tram phuong sai

def load_data_split(data_dir, max_images_per_emotion_train=800, max_images_per_emotion_test=200):
    """
    Gia su thu muc co dang
      fer2013/training/emotion
      fer2013/test/emotion
    """
    def load_subset(subdir, max_per_emotion):
        images = []
        labels = []
        base_path = os.path.join(data_dir, subdir)
        print(f"\nDang load du lieu tu {base_path}...")

        for idx, emotion in enumerate(EMOTIONS):
            emotion_path = os.path.join(base_path, emotion)
            if not os.path.exists(emotion_path):
                print(f"Khong tim thay thu muc: {emotion_path}")
                continue

            files = os.listdir(emotion_path)[:max_per_emotion]
            for f in files:
                img_path = os.path.join(emotion_path, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype('float32') / 255.0

                if img.std() < 0.02:
                    continue

                images.append(img.flatten())
                labels.append(idx)

            print(f"  {subdir} - {emotion}: {len([l for l in labels if l == idx])} anh")

        images = np.array(images)
        labels = np.array(labels)
        print(f"  Tong so anh {subdir}: {images.shape[0]}")
        return images, labels

    X_train, y_train = load_subset("training", max_images_per_emotion_train)
    X_test, y_test = load_subset("test", max_images_per_emotion_test)

    return X_train, y_train, X_test, y_test

def extract_hog_features(X_flat):
    print("\nTrich dac trung HOG...")
    hog_list = []
    n = X_flat.shape[0]
    for i in range(n):
        img = X_flat[i].reshape(IMG_SIZE)
        feat = hog(
            img,
            orientations=HOG_BINS,
            pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
            cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True
        )
        hog_list.append(feat)
        if (i + 1) % 500 == 0:
            print(f"  Da trich HOG {i + 1}/{n} anh...")

    hog_feat = np.array(hog_list)
    print(f"  HOG feature shape: {hog_feat.shape}")
    return hog_feat

def main():
    print("=" * 60)
    print("FER2013 - HOG + PCA + LinearSVC")
    print("=" * 60)

    # 1  Load du lieu train va test
    X_train, y_train, X_test, y_test = load_data_split(
        DATA_DIR,
        max_images_per_emotion_train=800,
        max_images_per_emotion_test=200
    )

    if X_train.size == 0 or X_test.size == 0:
        print("Du lieu bi rong. Kiem tra lai DATA_DIR va cau truc thu muc.")
        return

    # 2  Trich HOG
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # 3  Chuan hoa
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_hog)
    X_test_std = scaler.transform(X_test_hog)

    # 4  PCA
    print("\nAp dung PCA...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    print(f"  Shape train sau PCA: {X_train_pca.shape}")
    print(f"  Shape test sau PCA:  {X_test_pca.shape}")
    print(f"  Tong ti le phuong sai giai thich: {pca.explained_variance_ratio_.sum():.4f}")

    # 5  Train Linear SVM
    print("\nTraining LinearSVC...")
    clf = LinearSVC(C=1.0, random_state=42)
    clf.fit(X_train_pca, y_train)
    print("  Train xong.")

    # 6  Danh gia
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy tren tap test: {acc:.4f}")

    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

    # 7  Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS
    )
    plt.title('Confusion Matrix - HOG + PCA + LinearSVC')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix_hog_pca_svm.png", dpi=150)
    plt.close()
    print("\nDa luu confusion_matrix_hog_pca_svm.png trong thu muc outputs.")

if __name__ == "__main__":
    main()
