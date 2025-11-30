"""
Nguyen Thanh Dat - 211440519 - BTL
FER2013 CNN (train/test version)
- Doc anh tu folder training va test
- Dung CNN hoc dac trung truc tiep tren anh 48x48
- Danh gia tren tap test, ve confusion matrix va luu file
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# ==================== CAU HINH ====================
DATA_DIR = "fer2013"
TRAIN_DIR_NAME = "training"
TEST_DIR_NAME = "test"

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
N_CLASSES = len(EMOTIONS)
IMG_SIZE = (48, 48)

MAX_IMAGES_PER_EMOTION_TRAIN = 300
MAX_IMAGES_PER_EMOTION_TEST = 300

BATCH_SIZE = 64
EPOCHS = 20
FIG_DIR = "figs_cnn"

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
                if img.std() < 0.02:
                    continue
                images.append(img)
                labels.append(idx)
            except Exception:
                continue

        print(f"  {emotion}: {sum(np.array(labels) == idx)} anh")

    images = np.array(images)
    labels = np.array(labels)
    print(f"\nTong so anh load duoc: {images.shape[0]}")
    print(f"Input shape: {images.shape}")
    return images, labels

# ==================== 2. XAY DUNG CNN ====================
def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    # Fully connected
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

# ==================== MAIN ====================
def main():
    print("\n==============================================")
    print("FER2013 CNN - TRAIN/TEST VERSION")
    print("==============================================")

    os.makedirs(FIG_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, TRAIN_DIR_NAME)
    test_path = os.path.join(DATA_DIR, TEST_DIR_NAME)

    # Load tap TRAIN
    X_train_imgs, y_train = load_data(train_path, max_images_per_emotion=MAX_IMAGES_PER_EMOTION_TRAIN)
    if X_train_imgs.size == 0:
        print("Khong co du lieu TRAIN, thoat")
        return

    # Load tap TEST
    X_test_imgs, y_test = load_data(test_path, max_images_per_emotion=MAX_IMAGES_PER_EMOTION_TEST)
    if X_test_imgs.size == 0:
        print("Khong co du lieu TEST, thoat")
        return

    print(f"\nSo anh TRAIN: {len(X_train_imgs)}")
    print(f"So anh TEST:  {len(X_test_imgs)}")

    # Chuan hoa shape cho CNN: (N, 48, 48, 1)
    X_train = np.expand_dims(X_train_imgs, axis=-1)
    X_test = np.expand_dims(X_test_imgs, axis=-1)

    # Xay dung model
    model = build_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=N_CLASSES)

    # Train
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Danh gia tren TEST
    print("\n========== DANH GIA TREN TAP TEST ==========")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy (CNN): {test_acc:.4f}")
    print(f"Test loss (CNN):     {test_loss:.4f}")

    # Du doan
    y_test_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_prob, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix (CNN):")
    print(cm)

    # Ve confusion matrix
    cm_title = "CNN Confusion Matrix TEST"
    cm_path = os.path.join(FIG_DIR, "cnn_cm_test.png")
    plot_confusion_matrix(cm, EMOTIONS, title=cm_title, save_path=cm_path)

    # Accuracy per class va report
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, target_names=EMOTIONS))

    # In lai accuracy tong
    acc_final = accuracy_score(y_test, y_test_pred)
    print(f"\nAccuracy tong tren TAP TEST (CNN): {acc_final:.4f}")
    print(f"File anh confusion matrix CNN: {cm_path}")
    print("==============================================")

if __name__ == "__main__":
    # Co the fix random seed de lap lai ket qua
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
