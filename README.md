# Dự Án Dự Đoán Rủi Ro Bỏ Học (Dropout Prediction)

**Học phần:** Nhập môn Học máy — Trường Đại học Đại Nam  
**Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang  
**Nhóm thực hiện:** Nhóm 5 

### 👥 Danh sách thành viên
- **Phùng Quang Trung** – 1871020590
- **Đỗ Bảo Nguyên** – 1871020433
- **Đào Ngọc Nam** – 1871020404
- **Phạm Văn Tuấn Vũ** – 1871020672

---

## 📌 Mô tả chung
Dự án xây dựng quy trình học máy hoàn chỉnh (**End-to-End Machine Learning Pipeline**) nhằm dự đoán sinh viên có nguy cơ bỏ học dựa trên dữ liệu học tập và kinh tế – xã hội:
1. **Khám phá dữ liệu (EDA):** Kiểm tra kích thước, giá trị khuyết thiếu, trùng lặp và phân phối nhãn (`Graduate` vs `Dropout`).
2. **Tiền xử lý & Kỹ thuật đặc trưng (Preprocessing & Feature Engineering):** Chia tập Train/Test theo tỷ lệ 80/20 có phân tầng (`stratify`) để chống rò rỉ dữ liệu, triển khai 2 luồng xử lý:
   - **Luồng A (Baseline):** Điền khuyết cơ bản, mã hóa One-Hot và chuẩn hóa `StandardScaler`.
   - **Luồng B (Advanced):** Tạo biến phái sinh nghiệp vụ (tỷ lệ qua môn kỳ 1/kỳ 2, chênh lệch điểm, chỉ số rủi ro tài chính) kết hợp `RobustScaler`.
3. **Huấn luyện & Tinh chỉnh:** K-Nearest Neighbors (K-NN), Naive Bayes, Decision Tree và Random Forest với `GridSearchCV`.
4. **Đánh giá & Phân tích lỗi:** Đánh giá qua Accuracy, Precision, Recall, F1-Score, ROC-AUC; phân tầng rủi ro (Low / Medium / High) và phân tích ca lỗi FN/FP.
5. **Triển khai (Deployment):** Xây dựng REST API bằng **FastAPI** (`/health`, `/stats`, `/predict`) và giao diện **Dashboard** trực quan bằng Chart.js.

---

## 📂 Cấu trúc thư mục dự án

```text
DUDOANSINHVIEN/
├── README.md                    # Tài liệu hướng dẫn sử dụng dự án
├── requirements.txt             # Danh sách thư viện Python cần thiết
├── run_pipeline.py              # Script chạy toàn bộ pipeline huấn luyện & xuất kết quả
├── app.py                       # Điểm khởi chạy FastAPI server & Dashboard
│
├── data/
│   └── raw/                     # Thư mục chứa dữ liệu gốc (*.csv)
│
├── src/                         # Mã nguồn xử lý lõi
│   ├── __init__.py
│   ├── utils.py                 # Quản lý đường dẫn, random seed, phân tầng rủi ro
│   ├── domain_labels.py         # Ánh xạ mã ngành đào tạo
│   ├── preprocessing.py         # Đọc dữ liệu, làm sạch, EDA, chia tập, Luồng A & B
│   ├── train_evaluate.py        # Huấn luyện GridSearchCV, đánh giá, vẽ biểu đồ, lưu model
│   └── web_service.py           # Định nghĩa các endpoint FastAPI và phục vụ giao diện web
│
├── notebook/
│   └── Nhom5_complete.ipynb     # Jupyter Notebook trình bày chi tiết từng bước thực nghiệm
│
├── web/                         # Giao diện Frontend Dashboard
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── models/
│   └── best_dropout_pipeline.pkl # Pipeline mô hình tốt nhất (RandomForest - Flow B)
│
└── outputs/                     # Kết quả thực nghiệm & biểu đồ
    ├── Hinhve/                  # 7 biểu đồ PNG (EDA, ROC-AUC, Confusion Matrix, Feature Importance)
    ├── results_comparison.csv   # Bảng so sánh hiệu năng các mô hình trên Luồng A & B
    ├── tuning_results.csv       # Kết quả tìm siêu tham số tối ưu
    └── error_analysis.csv       # Chi tiết dự đoán, xác suất, mức rủi ro và phân loại lỗi FN/FP