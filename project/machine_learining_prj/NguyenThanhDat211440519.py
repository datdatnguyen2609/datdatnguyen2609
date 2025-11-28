# Danh gia Logistic Regression tren diabetes 

from sklearn.datasets import load_diabetes     # lay du lieu
from sklearn.model_selection import train_test_split  # chia train/test
from sklearn.linear_model import LogisticRegression   # mo hinh
import numpy as np                                  # tinh toan mang

# 1) Tai du lieu va nhi phan hoa
ds = load_diabetes()
X = ds.data
y = (ds.target >= np.median(ds.target)).astype(int)  # >= trung vi -> lop 1

# 2) Chia du lieu
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Tu chuan hoa (fit tren train, apply cho test)
mu = X_tr.mean(axis=0)                 # trung binh tren train
sigma = X_tr.std(axis=0, ddof=0)       # do lech chuan tren train
sigma[sigma == 0] = 1.0                # tranh chia 0
X_tr = (X_tr - mu) / sigma
X_te = (X_te - mu) / sigma

# 4) Huan luyen logistic
clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_tr)

# 5) Du doan
y_pred = clf.predict(X_te)                 # nhan 0/1
y_proba = clf.predict_proba(X_te)[:, 1]    # xac suat lop 1 (neu can)

# ===== CAC HAM TINH CHI SO THU CONG =====

def confusion_matrix_manual(y_true, y_hat):
    # tra ve tn, fp, fn, tp
    tn = fp = fn = tp = 0
    for yt, yp in zip(y_true, y_hat):
        if yt == 0 and yp == 0: tn += 1
        elif yt == 0 and yp == 1: fp += 1
        elif yt == 1 and yp == 0: fn += 1
        else: tp += 1
    return tn, fp, fn, tp

def accuracy_manual(tn, fp, fn, tp):
    return (tn + tp) / (tn + fp + fn + tp)

def precision_recall_f1(tp, fp, fn):
    # precision = tp/(tp+fp), recall = tp/(tp+fn), f1 = 2PR/(P+R)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1

def classification_report_manual(y_true, y_hat):
    tn, fp, fn, tp = confusion_matrix_manual(y_true, y_hat)
    # lop 1 (positive)
    p1, r1, f11 = precision_recall_f1(tp, fp, fn)
    # lop 0 (negative): hoan doi vai tro tp/fp/fn theo goc nhin lop 0
    tp0, fp0, fn0 = tn, fn, fp
    p0, r0, f10 = precision_recall_f1(tp0, fp0, fn0)
    # macro trung binh
    p_macro = (p0 + p1) / 2
    r_macro = (r0 + r1) / 2
    f1_macro = (f10 + f11) / 2
    acc = accuracy_manual(tn, fp, fn, tp)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": acc,
        "class_0": {"precision": p0, "recall": r0, "f1": f10},
        "class_1": {"precision": p1, "recall": r1, "f1": f11},
        "macro_avg": {"precision": p_macro, "recall": r_macro, "f1": f1_macro},
    }

# 6) Tinh chi so va in ket qua
metrics = classification_report_manual(y_te, y_pred)

print("Confusion matrix [[TN FP],[FN TP]]:")
print([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])

print("Accuracy:", round(metrics["accuracy"], 4))

print("Class 0 - precision:", round(metrics["class_0"]["precision"], 4),
      "recall:", round(metrics["class_0"]["recall"], 4),
      "f1:", round(metrics["class_0"]["f1"], 4))

print("Class 1 - precision:", round(metrics["class_1"]["precision"], 4),
      "recall:", round(metrics["class_1"]["recall"], 4),
      "f1:", round(metrics["class_1"]["f1"], 4))

print("Macro avg - precision:", round(metrics["macro_avg"]["precision"], 4),
      "recall:", round(metrics["macro_avg"]["recall"], 4),
      "f1:", round(metrics["macro_avg"]["f1"], 4))
