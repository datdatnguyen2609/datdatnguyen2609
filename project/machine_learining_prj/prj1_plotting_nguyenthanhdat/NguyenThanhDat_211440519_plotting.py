# Nguyen Thanh Dat - 211440519 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Doc file CSV (phan cach ;)
df = pd.read_csv("bangdiem_thcn1_k62.csv", sep=";", engine="python")

# Lam sach ten cot va bo cot rong
df.columns = df.columns.str.strip()
df = df.loc[:, [c for c in df.columns if c != ""]]

# Chuyen cac cot (tru "id") ve kieu so: thay "," -> "." ; gia tri loi -> NaN
num_cols = [c for c in df.columns if c.lower() != "id"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

# Chon cac cot diem (kieu so)
score_cols = [c for c in num_cols if np.issubdtype(df[c].dtype, np.number)]
if not score_cols:
    raise ValueError("Khong tim thay cot diem kieu so.")

# Tinh diem TB va phan loai
df["Diem TB"] = df[score_cols].mean(axis=1).round(2)
x = df["Diem TB"]
df["Phan loai"] = np.select(
    [(x>=5)&(x<7), (x>=7)&(x<8), (x>=8)&(x<9), (x>=9)&(x<=10)],
    ["Trung binh (5-7)", "Kha (7-8)", "Gioi (8-9)", "Xuat sac (9-10)"],
    default="Khac (<5 hoac >10)"
)

# Luu ket qua
df.to_csv("bangdiem_thcn1_k62_ketqua.csv", index=False, encoding="utf-8-sig")
print("Da luu: bangdiem_thcn1_k62_ketqua.csv")

# Ve bieu do 
plt.figure(figsize=(12,5)); df[score_cols].boxplot(); plt.title("Boxplot cac cot diem"); plt.xticks(rotation=45); plt.tight_layout(); plt.show()
plt.figure(figsize=(8,5));  plt.hist(df["Diem TB"].dropna(), bins=20); plt.title("Phan phoi diem TB"); plt.xlabel("Diem TB"); plt.ylabel("Tan suat"); plt.tight_layout(); plt.show()
order = ["Trung binh (5-7)","Kha (7-8)","Gioi (8-9)","Xuat sac (9-10)","Khac (<5 hoac >10)"]
counts = df["Phan loai"].value_counts().reindex(order, fill_value=0)
plt.figure(figsize=(8,5));  plt.bar(counts.index, counts.values); plt.title("So luong SV theo phan loai"); plt.ylabel("So luong"); plt.xticks(rotation=20); plt.tight_layout(); plt.show()
