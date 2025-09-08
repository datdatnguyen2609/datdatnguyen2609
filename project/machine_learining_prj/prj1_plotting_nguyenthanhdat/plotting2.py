import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV với dấu ; làm phân cách
file_path = "test01(BangDiem).csv"
df = pd.read_csv(file_path, delimiter=";")

# Bỏ cột STT vì không cần cho phân tích điểm
df_box = df.drop(columns=["STT"])

# Vẽ boxplot
plt.figure(figsize=(14, 6))
df_box.boxplot()
plt.title("Boxplot phân phối điểm các bài A1 - A20")
plt.xlabel("Bài kiểm tra")
plt.ylabel("Điểm")
plt.xticks(rotation=45)
plt.show()
