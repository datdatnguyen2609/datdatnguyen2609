"""
FER2013 Emotion Recognition - CNN Supervised Model
Su dung CNN co giam sat de cai thien accuracy so voi KMeans va SNN Izhikevich
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Cau hinh du lieu
DATA_DIR = "fer2013"
TRAIN_DIR = os.path.join(DATA_DIR, "training")
TEST_DIR  = os.path.join(DATA_DIR, "test")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
RANDOM_SEED = 42


def load_datasets():
    """
    Dung image_dataset_from_directory de load du lieu FER2013
    Gia su cau truc thu muc:
      fer2013/training/emotion
      fer2013/test/emotion
    """

    print("\nDang load du lieu training...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    print("\nDang load du lieu test...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Tach mot phan validation tu training
    val_size = 0.1
    val_batches = int(len(train_ds) * val_size)
    val_ds = train_ds.take(val_batches)
    train_ds2 = train_ds.skip(val_batches)

    # Cache va prefetch de train nhanh hon
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds2 = train_ds2.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds2, val_ds, test_ds


def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    """
    CNN don gian nhung kha manh cho FER2013
    """

    inputs = layers.Input(shape=input_shape)

    # Data augmentation nhe
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FER2013_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


    model.summary()
    return model


def train_model(model, train_ds, val_ds):
    """
    Train CNN voi early stopping va model checkpoint
    """
    os.makedirs("models", exist_ok=True)

    checkpoint_path = "models/fer2013_cnn_best.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Ve loss va accuracy
    os.makedirs("outputs", exist_ok=True)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train acc")
    plt.plot(epochs_range, val_acc, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train loss")
    plt.plot(epochs_range, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig("outputs/cnn_training_curves.png", dpi=150)
    plt.close()

    print("\nDa luu bieu do training tai outputs/cnn_training_curves.png")
    print(f"Model tot nhat duoc luu tai {checkpoint_path}")

    return model


def evaluate_model(model, test_ds):
    """
    Danh gia tren tap test va ve confusion matrix
    """

    print("\nDang danh gia tren tap test...")
    test_images = []
    test_labels = []

    for batch_x, batch_y in test_ds:
        test_images.append(batch_x.numpy())
        test_labels.append(batch_y.numpy())

    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)  # one hot

    y_true = np.argmax(test_labels, axis=1)

    y_prob = model.predict(test_images, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy CNN: {acc:.4f}")

    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix  FER2013 CNN")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix_cnn.png", dpi=150)
    plt.close()
    print("\nDa luu confusion_matrix_cnn.png trong outputs")

    return acc


def main():
    print("=" * 60)
    print("FER2013 Emotion Recognition  CNN Supervised Model")
    print("=" * 60)

    # 1  Load dataset
    train_ds, val_ds, test_ds = load_datasets()

    # 2  Xay dung model
    model = build_cnn_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1),
        num_classes=len(EMOTIONS)
    )

    # 3  Train
    model = train_model(model, train_ds, val_ds)

    # 4  Danh gia
    test_acc = evaluate_model(model, test_ds)

    print("\nHoan thanh. Test accuracy CNN =", test_acc)


if __name__ == "__main__":
    main()
