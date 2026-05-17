# Hướng Dẫn Chuẩn Bị Data Và Train Trên Kaggle

Tài liệu này giải thích flow chuẩn để chuẩn bị dữ liệu, upload lên Kaggle và chạy notebook train MobileNet/ResNet từ đầu.

## 1. Kết Luận Ngắn Gọn

Đúng: nên upload thư mục `data/cleaned/` lên Kaggle.

Notebook chính:

```text
notebooks/traffic_vehicle_classification_full_pipeline.ipynb
```

hiện được thiết kế để đọc dữ liệu từ:

```text
data/cleaned/<label>/
```

sau đó tự chia Train / Validation / Test theo tỉ lệ `70/15/15` bằng stratified split.

Vì vậy, khi chuẩn bị Kaggle dataset, ưu tiên upload:

```text
data/
└── cleaned/
    ├── bicycle/
    ├── boat/
    ├── bus/
    ├── car/
    ├── helicopter/
    ├── minibus/
    ├── motorcycle/
    ├── taxi/
    ├── train/
    └── truck/
```

## 2. Vậy Thư Mục `data/splits/` Dùng Để Làm Gì?

`data/splits/` là thư mục dùng khi bạn muốn chia dataset sẵn ở local bằng script:

```bash
python scripts/split_dataset.py --clear-output
```

Sau khi chạy, cấu trúc sẽ là:

```text
data/splits/
├── train/
│   ├── bicycle/
│   ├── boat/
│   └── ...
├── val/
│   ├── bicycle/
│   ├── boat/
│   └── ...
└── test/
    ├── bicycle/
    ├── boat/
    └── ...
```

Tuy nhiên, notebook hiện tại đang ưu tiên flow tự split từ `data/cleaned/`. Do đó:

- Nếu muốn đơn giản, upload `data/cleaned/`.
- Nếu muốn tái lập đúng split đã chia ở local, có thể upload thêm `data/splits/`, nhưng notebook cần chỉnh nhẹ để đọc trực tiếp `train/val/test`.
- Với yêu cầu báo cáo, tự split trong notebook là tốt vì notebook thể hiện rõ bước chia dữ liệu và kiểm tra phân bố class.

Khuyến nghị cho bài này: dùng `data/cleaned/` và để notebook tự split.

## 3. Flow Tổng Thể

```text
Crawl ảnh
  ↓
data/raw/<label>/
  ↓
Làm sạch ảnh lỗi, ảnh nhỏ, chuẩn hóa JPEG, đổi tên file
  ↓
data/cleaned/<label>/
  ↓
Lọc trùng bằng pHash/dHash + Hamming distance
  ↓
Review thủ công ảnh sai nhãn nếu cần
  ↓
Thống kê dataset
  ↓
Upload data/cleaned lên Kaggle
  ↓
Notebook tự split Train / Validation / Test
  ↓
Preprocess ảnh
  ↓
Train MobileNet from scratch
  ↓
Train ResNet from scratch
  ↓
Đánh giá và so sánh mô hình
```

## 4. Chuẩn Bị Data Ở Local

Chạy từ thư mục:

```bash
cd traffic_vehicle_classification
```

### Bước 1: Crawl Ảnh

Nếu muốn chạy tự động đến khi đủ 1500 ảnh sạch mỗi class:

```bash
python scripts/run_pipeline_until_ready.py --sources naver,duckduckgo --target-clean-per-label 1500 --batch-size 300
```

Log tiến độ:

```text
reports/pipeline_run_full.log
reports/pipeline_progress.csv
```

Theo dõi live:

```powershell
Get-Content reports\pipeline_run_full.log -Tail 80 -Wait
```

### Bước 2: Làm Sạch Ảnh

Nếu chạy pipeline tự động ở trên thì bước này đã được gọi tự động. Nếu chạy tay:

```bash
python scripts/remove_corrupted_images.py
```

Kết quả:

```text
data/cleaned/<label>/
reports/cleaning_report.csv
```

### Bước 3: Lọc Ảnh Trùng

```bash
python scripts/remove_duplicate_images.py --method phash --threshold 6 --action move
```

Logic:

- Tính pHash cho từng ảnh.
- So sánh Hamming distance giữa hash hiện tại và các hash đã giữ.
- Nếu distance `<= 6`, xem là ảnh trùng hoặc gần trùng.
- Ảnh trùng được chuyển sang `data/duplicates/<label>/`.

### Bước 4: Lọc Thủ Công

Tạo grid HTML để xem nhanh ảnh:

```bash
python scripts/manual_filter_helper.py --label car --max-per-label 120
```

Mở file:

```text
reports/manual_review.html
```

Các class nên review kỹ:

- `car`
- `taxi`
- `bus`
- `minibus`
- `truck`

vì các class này dễ bị lẫn ảnh.

### Bước 5: Thống Kê Dataset

```bash
python scripts/dataset_statistics.py
```

Kết quả:

```text
reports/dataset_summary.csv
reports/image_details.csv
reports/image_errors.csv
reports/figures/
```

Trước khi upload Kaggle, kiểm tra:

- Tổng số ảnh sạch > 10.000.
- Mỗi class nên có khoảng 1000 ảnh trở lên.
- Không class nào quá ít so với các class còn lại.

## 5. Chuẩn Bị File Zip Upload Lên Kaggle

Khuyến nghị nén folder `data/cleaned` thành một file zip để upload lên Kaggle.

Từ thư mục `traffic_vehicle_classification/`, cấu trúc trước khi nén nên là:

```text
data/
└── cleaned/
    ├── bicycle/
    ├── boat/
    ├── bus/
    ├── car/
    ├── helicopter/
    ├── minibus/
    ├── motorcycle/
    ├── taxi/
    ├── train/
    └── truck/
```

Trên Windows PowerShell, có thể nén bằng lệnh:

```powershell
Compress-Archive -Path data\cleaned -DestinationPath traffic_vehicle_cleaned.zip -Force
```

File zip nên có dạng:

```text
traffic_vehicle_cleaned.zip
└── cleaned/
    ├── bicycle/
    ├── boat/
    ├── bus/
    └── ...
```

Hoặc cũng được nếu zip có dạng:

```text
traffic_vehicle_cleaned.zip
└── data/
    └── cleaned/
        ├── bicycle/
        ├── boat/
        └── ...
```

Notebook hiện đã hỗ trợ cả hai dạng trên. Nếu Kaggle giữ nguyên file `.zip` trong `/kaggle/input`, notebook sẽ tự giải nén sang `/kaggle/working/extracted_datasets/`.

Không cần upload:

- `data/raw/` nếu chỉ để train.
- `data/duplicates/`.
- `data/rejected/`.
- `data/crawl_runs/`.

Nhưng nên giữ các file report để làm báo cáo:

```text
reports/dataset_summary.csv
reports/cleaning_report.csv
reports/duplicates.csv
reports/figures/
```

## 6. Cách Notebook Tìm Dataset Trên Kaggle

Trong Section 2 của notebook có biến:

```python
DATA_ROOT, DATA_MODE = find_dataset_paths()
```

Notebook tự tìm các path phổ biến như:

```text
/kaggle/input/traffic-vehicle-classification/data/cleaned
/kaggle/input/traffic-vehicle-classification/cleaned
/kaggle/input/*/data/cleaned
/kaggle/input/*/cleaned
/kaggle/input/**/*.zip
```

Nếu Kaggle Dataset của bạn có tên khác, notebook vẫn sẽ quét toàn bộ `/kaggle/input`. Nếu vẫn không tự nhận, chỉ cần sửa tay:

```python
DATA_ROOT = Path('/kaggle/input/<ten-dataset-cua-ban>/data/cleaned')
DATA_MODE = 'cleaned'
```

Ví dụ:

```python
DATA_ROOT = Path('/kaggle/input/vehicle-cleaned-dataset/data/cleaned')
DATA_MODE = 'cleaned'
```

## 7. Cấu Hình Train Trong Notebook

Ở Section 2, kiểm tra các biến:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RUN_TRAINING = True
MAX_IMAGES_PER_CLASS = None
EXPORT_SPLIT_FOLDERS = False
```

Ý nghĩa:

- `RUN_TRAINING = True`: bật train MobileNet và ResNet.
- `MAX_IMAGES_PER_CLASS = None`: dùng toàn bộ ảnh.
- `EXPORT_SPLIT_FOLDERS = False`: chỉ lưu split dataframe, không cần xuất folder split mới.

Nếu muốn test nhanh notebook trước khi train thật:

```python
MAX_IMAGES_PER_CLASS = 200
EPOCHS = 3
RUN_TRAINING = True
```

Khi train thật:

```python
MAX_IMAGES_PER_CLASS = None
EPOCHS = 30
RUN_TRAINING = True
```

## 8. Rà Soát Notebook Hiện Tại

Notebook hiện đã có đủ 13 section:

1. Introduction
2. Environment Setup
3. Dataset Loading
4. Dataset Overview and Statistics
5. Data Cleaning Summary
6. Train / Validation / Test Split
7. Image Preprocessing
8. Feature Extraction and Visualization
9. Model 1 - MobileNet From Scratch
10. Model 2 - ResNet From Scratch
11. Model Evaluation
12. Result Discussion
13. Conclusion

Notebook hiện dùng flow:

```text
DATA_ROOT = data/cleaned
↓
Tạo dataframe ảnh
↓
Tự stratified split train/val/test
↓
Tạo tf.data.Dataset
↓
Train MobileNet và ResNet from scratch
```

Notebook không dùng pretrained weights. Trong phần markdown có nhắc `weights='imagenet'` để nói rõ là không dùng, còn code model được xây từ các layer Keras cơ bản.

## 9. Lưu Ý Khi Train Trên Kaggle

- Bật GPU trong Kaggle Notebook: `Settings -> Accelerator -> GPU`.
- Nếu bộ dữ liệu lớn, lần đầu load ảnh có thể hơi chậm.
- Train từ đầu không dùng transfer learning nên cần nhiều epoch hơn và dữ liệu sạch hơn.
- Nếu accuracy chưa đạt 85%, ưu tiên cải thiện dữ liệu trước:
  - lọc sai nhãn,
  - tăng số ảnh class thiếu,
  - giảm ảnh trùng,
  - cân bằng class,
  - tăng augmentation hợp lý.

## 10. Nên Upload Gì?

Khuyến nghị upload lên Kaggle:

```text
data/cleaned/
reports/dataset_summary.csv
reports/figures/
```

Không bắt buộc upload:

```text
data/splits/
```

vì notebook đã tự split. Chỉ upload `data/splits/` nếu bạn muốn giữ split cố định đã chia ở local.
