# %% [markdown]
# # ĐỀ TÀI 37 – PHÂN TÍCH DỮ LIỆU LỖI PHẦN MỀM (Nhóm 2 – 523100B)
# **Hướng phân tích:** Thống kê + Phân loại (Decision Tree)
#
# **Bài toán:** dựa vào thông tin của một báo cáo lỗi, dự đoán lỗi có mức ưu tiên
# **Cao** (Blocker / Critical / Major) hay **Thấp** (Minor / Trivial).
#
# **Dữ liệu:** 200 lỗi của dự án HADOOP, lấy từ Apache Jira (https://issues.apache.org/jira),
# lưu trong file CSV.
#
# **Các bước (theo hướng dẫn):** đọc dữ liệu → kiểm tra → làm sạch → thống kê →
# thuật toán → đánh giá → biểu đồ → kết luận

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Tham số – đổi ở đây rồi chạy lại
filedulieu = "loi_phan_mem_nhom2_du_lieu_tho.csv"
tyletest = 0.25        # 25% dữ liệu để kiểm tra
maxdepth = 4           # độ sâu tối đa của cây
minsamplesleaf = 5     # số mẫu tối thiểu ở mỗi lá
criterion = "gini"     # "gini" hoặc "entropy"

# %% [markdown]
# ## Bước 1. Đọc dữ liệu

# %%
df = pd.read_csv(filedulieu)
print("Số dòng, số cột:", df.shape)
print(df.head())

# %% [markdown]
# ## Bước 2. Kiểm tra dữ liệu: thiếu – trùng – sai kiểu – không hợp lệ

# %%
print("Số giá trị thiếu mỗi cột:")
print(df.isnull().sum())

tieude = "Tiêu đề gốc (tiếng Anh)"
print("Số dòng trùng hoàn toàn:", df.duplicated().sum())
print("Số tiêu đề bị trùng    :", df[tieude].str.strip().duplicated().sum())

print("Kiểu dữ liệu cột ngày (đang là chuỗi → sai kiểu):")
print(df[["Ngày tạo", "Ngày xử lý xong"]].dtypes)

print("Số theo dõi / bình chọn bị âm:", ((df["Số người theo dõi"] < 0) | (df["Số lượt bình chọn"] < 0)).sum())

# %% [markdown]
# ## Bước 3. Làm sạch dữ liệu
# - Tiêu đề trùng → giữ 1 bản (một lỗi chỉ đếm một lần)
# - Cột ngày → đổi sang kiểu ngày tháng để tính số ngày xử lý
# - Thiếu *Cách giải quyết / Ngày xử lý* là do lỗi **chưa sửa** → không xóa, điền "Chưa giải quyết"
# - Thiếu *Module / Phiên bản* → điền "Không rõ" / rỗng (xóa sẽ mất ~20% dữ liệu)
# - Ngày xử lý trước ngày tạo → không hợp lệ, loại bỏ

# %%
df[tieude] = df[tieude].str.strip()
df = df.drop_duplicates(subset=tieude)

df["Ngày tạo"] = pd.to_datetime(df["Ngày tạo"])
df["Ngày xử lý xong"] = pd.to_datetime(df["Ngày xử lý xong"])

df["Cách giải quyết"] = df["Cách giải quyết"].fillna("Chưa giải quyết")
df["Thành phần (module)"] = df["Thành phần (module)"].fillna("Không rõ")
df["Phiên bản bị ảnh hưởng"] = df["Phiên bản bị ảnh hưởng"].fillna("")

df = df[~(df["Ngày xử lý xong"] < df["Ngày tạo"])].reset_index(drop=True)
print("Số dòng sau khi làm sạch:", len(df))

# %% [markdown]
# **Tạo thêm biến để phân tích.** Chỉ dùng thông tin có ngay lúc lỗi được báo;
# không dùng cột "Đặc trưng (lý do mức ưu tiên)" vì cột này viết dựa trên chính mức ưu tiên (rò rỉ dữ liệu).

# %%
df["songayxuly"] = (df["Ngày xử lý xong"] - df["Ngày tạo"]).dt.total_seconds() / 86400   # 86400 giây = 1 ngày
# Đếm số phần tử cách nhau bởi dấu ";"  (ví dụ "3.3.1; 3.4.0" → 2)
df["sophienban"] = df["Phiên bản bị ảnh hưởng"].str.count(";") + 1
df.loc[df["Phiên bản bị ảnh hưởng"] == "", "sophienban"] = 0
df["somodule"] = df["Thành phần (module)"].str.count(";") + 1
df.loc[df["Thành phần (module)"] == "Không rõ", "somodule"] = 0
df["dodaitieude"] = df[tieude].str.split().str.len()

module = df["Thành phần (module)"].str.split(";").str[0].str.strip()
df["modulechinh"] = module.where(module.map(module.value_counts()) >= 8, "Khác")

chu = df[tieude].str.lower()
df["cobaomat"] = chu.str.contains("cve|vulnerab|security|auth|token|kerberos|ssl").astype(int)
df["cocrash"] = chu.str.contains("npe|nullpointer|exception|fail|cannot|unable|leak|oom").astype(int)
df["cobuild"] = chu.str.contains("build|maven|release|jar|upgrade|bump").astype(int)
df["cotest"] = chu.str.contains("test|flaky").astype(int)

# Biến cần dự đoán: 1 = Cao, 0 = Thấp
df["muccao"] = df["Mức ưu tiên"].isin(["Chặn (Blocker)", "Nghiêm trọng (Critical)", "Lớn (Major)"]).astype(int)
print(df["muccao"].value_counts())

# %% [markdown]
# ## Bước 4. Thống kê: mean, median, variance, std, correlation, ngoại lệ

# %%
bienso = ["songayxuly", "Số người theo dõi", "Số lượt bình chọn", "sophienban", "somodule", "dodaitieude"]
print(df[bienso].agg(["mean", "median", "var", "std", "min", "max"]).round(2).T)

# Ngoại lệ theo IQR: lớn hơn Q3 + 1.5*IQR
songay = df["songayxuly"].dropna()
q1, q3 = songay.quantile(0.25), songay.quantile(0.75)
nguong = q3 + 1.5 * (q3 - q1)
print(f"Ngưỡng ngoại lệ: {nguong:.0f} ngày → có {(songay > nguong).sum()} lỗi xử lý lâu bất thường (giữ lại, không xóa)")

print("Tỷ lệ % mức Cao theo loại lỗi:")
print((df.groupby("Loại lỗi")["muccao"].mean() * 100).round(1).sort_values())

tuongquan = df[bienso + ["cobaomat", "cocrash", "cobuild", "cotest", "muccao"]].corr()
print("Tương quan với muccao:")
print(tuongquan["muccao"].drop("muccao").round(3).sort_values())

# %% [markdown]
# ## Bước 5. Thuật toán: Cây quyết định (Decision Tree)
# - Cây hỏi lần lượt các câu "đặc trưng ≤ ngưỡng?" để chia lỗi thành 2 nhánh.
# - Chọn câu hỏi làm **độ vẩn đục Gini** giảm nhiều nhất: $Gini = 1 - \sum p_k^2$
#   (0 = nút chỉ có 1 lớp, 0.5 = hai lớp chia đều).
# - Dừng khi cây đạt `maxdepth` hoặc lá có ít hơn `minsamplesleaf` mẫu; lá dự đoán theo lớp đa số.
# - `class_weight="balanced"`: tăng trọng số lớp Thấp vì lớp này chỉ chiếm ~25%.

# %%
dactrung = ["Số người theo dõi", "Số lượt bình chọn", "sophienban", "somodule", "dodaitieude",
            "cobaomat", "cocrash", "cobuild", "cotest", "Loại lỗi", "modulechinh"]
x = pd.get_dummies(df[dactrung], columns=["Loại lỗi", "modulechinh"], prefix_sep=" = ", dtype=int)
y = df["muccao"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=tyletest, random_state=42, stratify=y)

cay = DecisionTreeClassifier(max_depth=maxdepth, min_samples_leaf=minsamplesleaf, criterion=criterion,
                             class_weight="balanced", random_state=42)
cay.fit(xtrain, ytrain)
ydudoan = cay.predict(xtest)

# %% [markdown]
# ## Bước 6. Đánh giá: Accuracy, Precision, Recall, F1
# Dữ liệu mất cân bằng nên so sánh với **Baseline** (luôn đoán "Cao") và xem thêm **F1-macro**
# (trung bình F1 của cả 2 lớp). Dữ liệu ít nên kiểm tra thêm bằng **kiểm định chéo 5 lần**.

# %%
print("Accuracy :", round(accuracy_score(ytest, ydudoan), 3))
print("Precision:", round(precision_score(ytest, ydudoan), 3))
print("Recall   :", round(recall_score(ytest, ydudoan), 3))
print("F1       :", round(f1_score(ytest, ydudoan), 3))
print("F1-macro :", round(f1_score(ytest, ydudoan, average="macro"), 3))
print("Ma trận nhầm lẫn (hàng = thực tế Thấp/Cao, cột = dự đoán Thấp/Cao):")
print(confusion_matrix(ytest, ydudoan))

nam = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # chia 5 phần, giữ tỷ lệ Cao/Thấp
baseline = DummyClassifier(strategy="most_frequent")
print("Kiểm định chéo 5 lần (F1-macro):")
print("  Decision Tree:", cross_val_score(cay, x, y, cv=nam, scoring="f1_macro").mean().round(3))
print("  Baseline     :", cross_val_score(baseline, x, y, cv=nam, scoring="f1_macro").mean().round(3))

# %% [markdown]
# ## Bước 7. Biểu đồ

# %%
# Biểu đồ 1 – Bar chart: số lỗi theo mức ưu tiên (so sánh các nhóm)
thutu = ["Chặn (Blocker)", "Nghiêm trọng (Critical)", "Lớn (Major)", "Nhỏ (Minor)", "Không đáng kể (Trivial)"]
dem = df["Mức ưu tiên"].value_counts().reindex(thutu)
plt.figure(figsize=(8, 4))
plt.bar(dem.index, dem.values, color="#4472c4")
plt.title("Biểu đồ 1. Số lỗi theo mức ưu tiên")
plt.ylabel("Số lỗi")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# *Major chiếm đa số (121/199 lỗi) → dữ liệu mất cân bằng, nhóm Cao chiếm ~75%.*

# %%
# Biểu đồ 2 – Histogram: phân phối số ngày xử lý
plt.figure(figsize=(8, 4))
plt.hist(songay, bins=20, color="#70ad47", edgecolor="white")
plt.axvline(songay.mean(), color="red", linestyle="--", label=f"Mean = {songay.mean():.0f}")
plt.axvline(songay.median(), color="black", linestyle="--", label=f"Median = {songay.median():.0f}")
plt.title("Biểu đồ 2. Phân phối số ngày xử lý lỗi")
plt.xlabel("Số ngày")
plt.ylabel("Số lỗi")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# *Phân phối lệch phải: đa số lỗi sửa trong vài tuần, một số kéo dài hàng năm → Mean (73) lớn hơn
# nhiều so với Median (11) → dùng Median để mô tả thời gian xử lý điển hình.*

# %%
# Biểu đồ 3 – Heatmap: tương quan giữa các biến
plt.figure(figsize=(9, 7))
sns.heatmap(tuongquan, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Biểu đồ 3. Ma trận tương quan")
plt.tight_layout()
plt.show()

# %% [markdown]
# *Hàng/cột `muccao` đều gần 0 (|r| ≤ 0.13) → không biến nào liên hệ mạnh với mức ưu tiên.*

# %%
# Biểu đồ 4 – Sơ đồ cây quyết định (đọc được luật "nếu… thì…")
plt.figure(figsize=(20, 9))
plot_tree(cay, feature_names=x.columns, class_names=["Thấp", "Cao"], filled=True, rounded=True, fontsize=8)
plt.title("Biểu đồ 4. Cây quyết định phân loại mức ưu tiên")
plt.tight_layout()
plt.show()

# %% [markdown]
# *Câu hỏi đầu tiên: "Lỗi có phải loại Tài liệu không?" – nếu có → Thấp. Đây là yếu tố quan trọng nhất.*
#
# ## Bước 8. Kết luận
# 1. Mức ưu tiên lệch mạnh về "Major" (~75% lỗi mức Cao); thời gian xử lý lệch phải, nhiều ngoại lệ.
# 2. Không có yếu tố nào liên hệ mạnh với mức ưu tiên (tương quan đều rất yếu); dấu hiệu rõ nhất là
#    lỗi Tài liệu thường mức Thấp.
# 3. Trên tập test, cây đạt F1-macro 0.51 (Baseline 0.43); nhưng kiểm định chéo cho cây ≈ 0.43, gần bằng Baseline
#    → chỉ với các thông tin này thì **khó dự đoán chính xác** mức ưu tiên. Cây vẫn có ích để **giải thích**
#    (luật "nếu… thì…").
# 4. **Hạn chế:** dữ liệu ít (199 lỗi, 1 dự án), mức ưu tiên do con người gán nên chủ quan.
#    **Hướng phát triển:** thêm dữ liệu, dùng nội dung mô tả lỗi (TF-IDF), thử mô hình khác.
