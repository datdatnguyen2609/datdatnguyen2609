"""
FER2013 Emotion Recognition - SNN Izhikevich Reservoir + LinearSVC
Su dung SNN (neuron Izhikevich) lam reservoir de cai thien accuracy so voi KMeans
"""

import os
import numpy as np
import cv2

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Cau hinh du lieu
DATA_DIR = "fer2013"
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (48, 48)

# Cau hinh PCA
PCA_COMPONENTS = 64     # so chieu sau PCA (ban co the thu 32, 64, 128)

# Cau hinh SNN reservoir
N_HIDDEN = 100          # so neuron Izhikevich trong reservoir
SIM_STEPS = 20          # so time step mo phong SNN
INPUT_RATE = 20.0       # max firing rate Hz tu feature  Poisson encoding
DT = 1.0                # buoc thoi gian gia su = 1 ms

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ===================== 1  LOAD DU LIEU =====================

def load_data_split(data_dir,
                    max_images_per_emotion_train=800,
                    max_images_per_emotion_test=200):
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

                # Loc bot anh rat xau
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

# ===================== 2  PCA TIEN XU LY =====================

def pca_preprocess(X_train, X_test, n_components=PCA_COMPONENTS):
    print("\nChuan hoa va giam chieu bang PCA...")
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    print(f"  Shape train sau PCA: {X_train_pca.shape}")
    print(f"  Shape test  sau PCA: {X_test_pca.shape}")
    print(f"  Tong ti le phuong sai giai thich: {pca.explained_variance_ratio_.sum():.4f}")
    return X_train_pca, X_test_pca

# ===================== 3  MA HOA SPIKE  POISSON =====================

def poisson_encode(features, sim_steps=SIM_STEPS, max_rate=INPUT_RATE):
    """
    features: (N, D), da scale ve [0, 1]
    Tra ve: spikes (N, sim_steps, D)  gia tri 0 hoac 1
    """
    N, D = features.shape
    # Xem feature nhu rate trong [0, 1], nhan voi max_rate, dt=1
    rates = features * max_rate  # Hz
    # xac suat spike moi time step = rate * dt / 1000 neu dt = 1 ms
    p_spike = rates * (DT / 1000.0)
    p_spike = np.clip(p_spike, 0.0, 1.0)

    spikes = np.random.rand(N, sim_steps, D) < p_spike[:, None, :]
    return spikes.astype(np.float32)

# ===================== 4  SNN RESERVOIR VOI IZHIKVICH =====================

class IzhikevichReservoir:
    """
    Reservoir gom N_HIDDEN neuron Izhikevich
    Tron so dau vao random, khong train
    Doc ra bang tong so spike cua moi neuron trong khoang thoi gian
    """

    def __init__(self,
                 input_dim,
                 n_hidden=N_HIDDEN,
                 dt=DT):
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.dt = dt

        # Tron so dau vao random, scale nho de tranh no
        self.W_in = np.random.normal(loc=0.0, scale=0.5,
                                     size=(n_hidden, input_dim)).astype(np.float32)

        # Tham so Izhikevich  RS  regular spiking
        self.a = 0.02 * np.ones(n_hidden, dtype=np.float32)
        self.b = 0.2 * np.ones(n_hidden, dtype=np.float32)
        self.c = -65.0 * np.ones(n_hidden, dtype=np.float32)
        self.d = 8.0 * np.ones(n_hidden, dtype=np.float32)

        # Trang thai ban dau
        self.v0 = -65.0 * np.ones(n_hidden, dtype=np.float32)
        self.u0 = self.b * self.v0

    def run_sample(self, spike_train):
        """
        spike_train: shape (T, D)  D = input_dim
        Tra ve: spike_counts shape (n_hidden,)
        """
        T, D = spike_train.shape
        assert D == self.input_dim

        v = self.v0.copy()
        u = self.u0.copy()

        spike_counts = np.zeros(self.n_hidden, dtype=np.float32)

        for t in range(T):
            x_t = spike_train[t]  # (D,)
            I_t = self.W_in @ x_t  # (n_hidden,)

            # Mo phong Izhikevich  discrete time
            # dv/dt = 0.04v^2 + 5v + 140 - u + I
            # du/dt = a(bv - u)
            # Su dung 2 buoc Euler dt/2 de on dinh hon  Izhikevich goi la "0.5 step"
            for _ in range(2):
                dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I_t
                du = self.a * (self.b * v - u)
                v += dv * (self.dt * 0.5)
                u += du * (self.dt * 0.5)

            # Spike va reset
            fired = v >= 30.0
            spike_counts[fired] += 1.0
            v[fired] = self.c[fired]
            u[fired] += self.d[fired]

        return spike_counts

    def transform(self, spike_trains):
        """
        spike_trains: shape (N, T, D)
        Tra ve: features_snn shape (N, n_hidden)
        """
        N, T, D = spike_trains.shape
        print(f"\nDang mo phong reservoir Izhikevich cho {N} mau...")
        feats = []
        for i in range(N):
            sc = self.run_sample(spike_trains[i])
            feats.append(sc)
            if (i + 1) % 200 == 0:
                print(f"  Da xu ly {i + 1}/{N} mau...")

        feats = np.stack(feats, axis=0)  # (N, n_hidden)
        # Co the chia cho T de tro thanh firing rate
        feats = feats / float(T)
        print(f"  Feature SNN shape: {feats.shape}")
        return feats

# ===================== 5  TRAIN CLASSIFIER =====================

def train_and_evaluate_classifier(X_train_snn, y_train, X_test_snn, y_test):
    print("\nTraining LinearSVC tren feature SNN...")
    clf = LinearSVC(C=1.0, random_state=RANDOM_SEED)
    clf.fit(X_train_snn, y_train)
    print("  Train xong.")

    y_pred = clf.predict(X_test_snn)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy tren tap test  SNN Izhikevich + LinearSVC: {acc:.4f}")

    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

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
    plt.title('Confusion Matrix - SNN Izhikevich Reservoir + LinearSVC')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix_snn_izhikevich.png", dpi=150)
    plt.close()
    print("\nDa luu outputs/confusion_matrix_snn_izhikevich.png")

# ===================== MAIN =====================

def main():
    print("=" * 60)
    print("FER2013 - SNN Izhikevich Reservoir + LinearSVC")
    print("=" * 60)

    # 1  Load du lieu
    X_train, y_train, X_test, y_test = load_data_split(
        DATA_DIR,
        max_images_per_emotion_train=800,
        max_images_per_emotion_test=200
    )

    if X_train.size == 0 or X_test.size == 0:
        print("Du lieu bi rong. Kiem tra lai DATA_DIR va cau truc thu muc.")
        return

    # 2  PCA
    X_train_pca, X_test_pca = pca_preprocess(X_train, X_test, n_components=PCA_COMPONENTS)

    # 3  Scale ve [0, 1] de dung lam rate cho Poisson
    mm_scaler = MinMaxScaler()
    X_train_scaled = mm_scaler.fit_transform(X_train_pca)
    X_test_scaled = mm_scaler.transform(X_test_pca)

    # 4  Ma hoa thanh spike train
    print("\nMa hoa Poisson spike train tu feature PCA...")
    train_spikes = poisson_encode(X_train_scaled)
    test_spikes = poisson_encode(X_test_scaled)
    print(f"  train_spikes shape: {train_spikes.shape}")
    print(f"  test_spikes  shape: {test_spikes.shape}")

    # 5  Reservoir SNN Izhikevich
    reservoir = IzhikevichReservoir(input_dim=X_train_scaled.shape[1],
                                    n_hidden=N_HIDDEN,
                                    dt=DT)

    X_train_snn = reservoir.transform(train_spikes)
    X_test_snn = reservoir.transform(test_spikes)

    # 6  Train classifier va danh gia
    train_and_evaluate_classifier(X_train_snn, y_train, X_test_snn, y_test)

    print("\nHoan thanh pipeline SNN Izhikevich.")

if __name__ == "__main__":
    main()
