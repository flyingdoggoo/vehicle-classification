# Hướng Dẫn Đưa Dataset Lên Nền Tảng Online Để Lọc Lại Lần Cuối

## 1. Nên Chọn Nền Tảng Nào?

Khuyến nghị dùng **Roboflow** cho project này.

Lý do:

- Dữ liệu của mình là bài toán **image classification**, tức mỗi ảnh có một nhãn cấp ảnh như `car`, `bus`, `taxi`.
- Roboflow hỗ trợ project classification, trong đó classification là **image-level labels only**, không cần vẽ bounding box.
- Roboflow có giao diện web dễ dùng, phù hợp để mời nhiều bạn cùng vào kiểm tra ảnh sai nhãn.
- Có thể upload dataset lớn bằng command line, phù hợp dataset hơn 10.000 ảnh.
- Sau khi review xong có thể export/download lại dataset để train trên Kaggle.

Lựa chọn thay thế:

- **Label Studio**: tốt nếu muốn tự host, kiểm soát dữ liệu riêng tư hơn, nhưng setup phức tạp hơn.
- **CVAT**: mạnh cho detection/segmentation, nhưng với bài này chỉ cần classification nên hơi nặng.
- **FiftyOne**: rất tốt để khám phá/lọc dữ liệu local, nhưng không tiện bằng Roboflow nếu muốn nhiều người vào web review cùng lúc.

Kết luận thực tế:

```text
Muốn nhanh, dễ mời người review: Roboflow
Muốn tự host/private hơn: Label Studio
Muốn phân tích/lọc local nâng cao: FiftyOne
```

## 2. Lưu Ý Về Quyền Riêng Tư

Nếu dùng Roboflow, cần chú ý workspace/project public hay private.

Với dữ liệu crawl từ web phục vụ học tập, public workspace có thể chấp nhận được. Nhưng nếu có ảnh nhạy cảm hoặc dữ liệu riêng, nên dùng workspace private hoặc tự host Label Studio.

## 3. Chuẩn Bị Dataset Trước Khi Upload

Chỉ upload dữ liệu đã clean:

```text
data/cleaned/
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

Không nên upload:

```text
data/raw/
data/duplicates/
data/rejected/
data/crawl_runs/
```

Vì:

- `raw` còn nhiều ảnh lỗi hoặc nhiễu.
- `duplicates` là ảnh đã bị loại.
- `rejected` là ảnh sai/không đạt yêu cầu.
- `crawl_runs` chỉ là folder tạm/log crawl.

## 4. Flow Review Khuyến Nghị Trên Roboflow

```text
data/cleaned ở local
  ↓
Tạo project Roboflow loại Single-Label Classification
  ↓
Upload ảnh theo folder label
  ↓
Mời thành viên vào workspace/project
  ↓
Mỗi người review một nhóm class
  ↓
Ảnh đúng nhãn: giữ lại
  ↓
Ảnh sai nhãn: sửa nhãn hoặc xóa khỏi dataset
  ↓
Ảnh mơ hồ/không thấy phương tiện rõ: xóa khỏi dataset
  ↓
Export/download dataset đã review
  ↓
Đưa dataset đã review vào Kaggle để train
```

## 5. Cách Tạo Project Roboflow

1. Vào Roboflow và đăng nhập.
2. Tạo workspace hoặc dùng workspace có sẵn.
3. Tạo project mới.
4. Chọn project type:

```text
Single-Label Classification
```

5. Tạo các class:

```text
bicycle
boat
bus
car
helicopter
minibus
motorcycle
taxi
train
truck
```

6. Upload ảnh từ `data/cleaned/`.

Với dataset nhỏ, có thể kéo thả trên web. Với dataset lớn hơn 10.000 ảnh, nên upload bằng command line.

## 6. Upload Dataset Lớn Bằng Roboflow CLI

Cài Roboflow Python package:

```bash
pip install roboflow
```

Sau khi tạo project trên Roboflow, lấy:

- Workspace ID
- Project ID
- API key

Sau đó upload folder dataset:

```bash
roboflow import -w <workspace-id> -p <project-id> data/cleaned
```

Nếu CLI yêu cầu API key, đăng nhập hoặc cấu hình theo hướng dẫn trong Roboflow.

## 7. Cách Chia Việc Cho Nhóm Review

Nên chia theo class:

```text
Người 1: bicycle, motorcycle
Người 2: car, taxi
Người 3: bus, minibus, truck
Người 4: boat, train, helicopter
```

Các class cần review kỹ:

- `car` và `taxi`: taxi có thể bị nhầm thành car thường.
- `bus` và `minibus`: kích thước/góc chụp dễ giống nhau.
- `truck` và `bus`: ảnh góc ngang hoặc xe lớn dễ nhầm.
- `boat`: dễ lẫn ảnh phong cảnh nước nhưng không rõ boat.
- `train`: dễ lẫn ảnh ga tàu hoặc đường ray không có train.
- `helicopter`: dễ lẫn ảnh đồ chơi/mô hình/ảnh icon.

## 8. Quy Tắc Giữ/Xóa/Sửa Nhãn

Giữ ảnh nếu:

- Phương tiện chính nhìn rõ.
- Ảnh thuộc đúng class.
- Ảnh không quá mờ, không quá nhỏ.
- Không phải ảnh vẽ/icon/logo nếu bạn muốn train ảnh thực tế.

Sửa nhãn nếu:

- Ảnh rõ phương tiện nhưng đang nằm sai class.
- Ví dụ ảnh taxi nằm trong `car`, hoặc minibus nằm trong `bus`.

Xóa ảnh nếu:

- Không có phương tiện.
- Ảnh quá mờ hoặc quá nhỏ.
- Ảnh chỉ là logo/icon/đồ họa không giống ảnh thực tế.
- Ảnh có quá nhiều phương tiện mà không xác định được class chính.
- Ảnh bị watermark che nặng.
- Ảnh trùng hoặc gần trùng.

## 9. Sau Khi Review Xong

Export/download dataset đã review.

Cấu trúc mong muốn sau khi tải về:

```text
reviewed_dataset/
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

Nếu Roboflow export ra cấu trúc khác, cần chuyển lại về folder-per-class trước khi đưa vào Kaggle.

Notebook Kaggle hiện cần:

```text
data/cleaned/<label>/*.jpg
```

## 10. Upload Lên Kaggle Sau Review

Upload dataset đã review lên Kaggle theo cấu trúc:

```text
traffic-vehicle-classification-reviewed/
└── data/
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

Trong notebook, nếu cần chỉnh path:

```python
DATA_ROOT = Path('/kaggle/input/traffic-vehicle-classification-reviewed/data/cleaned')
```

Notebook sẽ tự split train/validation/test, nên không bắt buộc upload `data/splits/`.

## 11. Checklist Trước Khi Train

Trước khi train trên Kaggle, kiểm tra:

- Tổng ảnh sạch > 10.000.
- Mỗi class có đủ số lượng tương đối cân bằng.
- Không còn class bị thiếu quá nhiều.
- Các class dễ nhầm đã được review thủ công.
- `reports/dataset_summary.csv` đã được cập nhật.
- Notebook Section 2 trỏ đúng `DATA_ROOT`.

## 12. Nguồn Tham Khảo

- Roboflow Annotate: https://docs.roboflow.com/annotate/annotation-tools
- Roboflow Upload Data: https://docs.roboflow.com/datasets/adding-data
- Label Studio Overview: https://labelstud.io/guide/get_started
- CVAT image annotation/classification overview: https://www.cvat.ai/resources/blog/image-annotation-guide
