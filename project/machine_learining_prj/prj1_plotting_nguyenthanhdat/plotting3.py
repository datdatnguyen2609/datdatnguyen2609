import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Doc file CSV (phan cach ;) va loai bo cot trong
df = pd.read_csv("bangdiem_thcn1_k62.csv", sep=";", engine="python")
df.columns = [c.strip() for c in df.columns]
df = df.loc[:, df.columns.astype(str).str.strip() != ""]   # bo cot ten rong

# Chuyen so: thay "," -> "." roi to_numeric; giu cot ID neu co
for c in df.columns:
    if c.lower() != "id":
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

# Chon cot diem (tat ca cot so, tru ID neu co)
score_cols = [c for c in df.columns if c.lower() != "id" and np.issubdtype(df[c].dtype, np.number)]
if not score_cols:
    raise ValueError("Khong tim thay cot diem.")

# Diem trung binh va phan loai
diem_tb = df[score_cols].mean(axis=1, skipna=True)

def phan_loai(x):
    if 5 <= x < 7:  return "Trung binh (5-7)"
    if 7 <= x < 8:  return "Kha (7-8)"
    if 8 <= x < 9:  return "Gioi (8-9)"
    if 9 <= x <=10: return "Xuat sac (9-10)"
    return "Khac (<5 hoac >10)"

df["Diem TB"] = diem_tb.round(2)
df["Phan loai"] = df["Diem TB"].apply(phan_loai)
df.to_csv("bangdiem_thcn1_k62_ketqua.csv", index=False)
print("Da luu: bangdiem_thcn1_k62_ketqua.csv")

# Ve nhanh 3 bieu do
plt.figure(figsize=(12,5)); df[score_cols].boxplot(); plt.title("Boxplot cac cot diem"); plt.xticks(rotation=45); plt.tight_layout(); plt.show()
plt.figure(figsize=(8,5));  plt.hist(df["Diem TB"].dropna(), bins=20); plt.title("Phan phoi diem TB"); plt.tight_layout(); plt.show()
counts = df["Phan loai"].value_counts().reindex(["Trung binh (5-7)","Kha (7-8)","Gioi (8-9)","Xuat sac (9-10)","Khac (<5 hoac >10)"], fill_value=0)
plt.figure(figsize=(8,5));  plt.bar(counts.index, counts.values); plt.title("So luong SV theo phan loai"); plt.xticks(rotation=20); plt.tight_layout(); plt.show()
