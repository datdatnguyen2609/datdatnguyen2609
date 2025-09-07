import matplotlib.pyplot as plt
import pandas as pd

# Doc diem tu file csdl.txt, moi dong la mot diem
with open('csdl.txt', 'r', encoding='utf-8') as f:
    diem_qua_trinh = []
    for line in f:
        try:
            diem = float(line.strip().replace(",", "."))
            diem_qua_trinh.append(diem)
        except ValueError:
            pass  # Bo qua dong loi neu co

# Chuyen sang DataFrame de de thong ke
df = pd.DataFrame({'Diem qua trinh': diem_qua_trinh})

# Hien thi bang diem
print("Bang diem:")
print(df)

# Thong ke mo ta (describe)
print("\nThong ke mo ta (describe):")
print(df.describe())

# Thong ke tan suat diem (value_counts)
print("\nThong ke tan suat diem (value_counts):")
print(df['Diem qua trinh'].value_counts().sort_index())

# Thong ke phan nhom (theo khoang diem)
bins = [0, 4, 5, 6, 7, 8, 9, 10]
labels = ['0-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
df['Nhom diem'] = pd.cut(df['Diem qua trinh'], bins=bins, labels=labels, right=True, include_lowest=True)
print("\nThong ke theo nhom diem:")
print(df['Nhom diem'].value_counts().sort_index())

# Thong ke chi tiet: trung binh, trung vi, mode, phuong sai, do lech chuan, min/max
print("\nThong ke chi tiet:")
print(f"Diem trung binh (mean): {df['Diem qua trinh'].mean():.2f}")
print(f"Trung vi (median): {df['Diem qua trinh'].median():.2f}")
print(f"Mode (diem xuat hien nhieu nhat): {df['Diem qua trinh'].mode().tolist()}")
print(f"Phuong sai (variance): {df['Diem qua trinh'].var():.2f}")
print(f"Do lech chuan (std): {df['Diem qua trinh'].std():.2f}")
print(f"Diem nho nhat (min): {df['Diem qua trinh'].min()}")
print(f"Diem lon nhat (max): {df['Diem qua trinh'].max()}")

# Thong ke phan tram so diem theo nhom
print("\nPhan tram so diem theo nhom:")
print(df['Nhom diem'].value_counts(normalize=True).sort_index().apply(lambda x: f"{100*x:.2f}%"))

# Ve histogram (phan bo diem)
plt.figure(figsize=(8,5))
plt.hist(df['Diem qua trinh'], bins=10, color='green', edgecolor='black')
plt.title("Phan bo diem qua trinh mon co so du lieu")
plt.xlabel("Diem")
plt.ylabel("So luong")
plt.tight_layout()
plt.show()

# Ve boxplot (bieu do hop)
plt.figure(figsize=(7,5))
plt.boxplot(df['Diem qua trinh'], vert=False)
plt.title("Boxplot diem qua trinh mon co so du lieu")
plt.xlabel("Diem")
plt.tight_layout()
plt.show()