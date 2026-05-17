# Hướng Dẫn Lọc Dataset Bằng Google Drive

Tài liệu này dùng cho bước review thủ công cuối cùng sau khi đã crawl, clean và lọc trùng dữ liệu ở local.

## 1. Khi Nào Nên Dùng Google Drive?

Nên dùng Google Drive nếu mục tiêu là:

- Xem lại ảnh theo từng class.
- Xóa ảnh sai, ảnh mờ, ảnh không có phương tiện.
- Chuyển ảnh sai nhãn sang đúng folder.
- Chia cho nhiều người cùng lọc nhanh.

Với bài toán này, Google Drive là đủ vì đây là bài toán **image classification**. Mỗi ảnh chỉ cần một nhãn cấp ảnh, ví dụ `car`, `bus`, `taxi`, không cần vẽ bounding box.

## 2. Nên Upload Folder Nào?

Upload folder đã clean:

```text
traffic_vehicle_classification/data/cleaned/
```

Không upload các folder sau để review:

```text
data/raw/
data/duplicates/
data/crawl_runs/
data/splits/
```

Lý do:

- `raw` còn ảnh lỗi, ảnh nhiễu, ảnh chưa chuẩn hóa.
- `duplicates` là ảnh đã bị coi là trùng hoặc gần trùng.
- `crawl_runs` chỉ là dữ liệu tạm trong quá trình crawl.
- `splits` chỉ nên tạo sau khi dataset cuối cùng đã được review xong.

## 3. Cấu Trúc Nên Tạo Trên Google Drive

Tạo một folder trên Google Drive như sau:

```text
traffic_vehicle_review/
├── cleaned_for_review/
│   ├── bicycle/
│   ├── boat/
│   ├── bus/
│   ├── car/
│   ├── helicopter/
│   ├── minibus/
│   ├── motorcycle/
│   ├── taxi/
│   ├── train/
│   └── truck/
│
├── rejected/
│   ├── bicycle/
│   ├── boat/
│   ├── bus/
│   ├── car/
│   ├── helicopter/
│   ├── minibus/
│   ├── motorcycle/
│   ├── taxi/
│   ├── train/
│   └── truck/
│
└── manual_review/
    ├── relabel_to_bicycle/
    ├── relabel_to_boat/
    ├── relabel_to_bus/
    ├── relabel_to_car/
    ├── relabel_to_helicopter/
    ├── relabel_to_minibus/
    ├── relabel_to_motorcycle/
    ├── relabel_to_taxi/
    ├── relabel_to_train/
    └── relabel_to_truck/
```

Trong đó:

- `cleaned_for_review/`: dữ liệu chính để mọi người lọc.
- `rejected/`: ảnh bị loại khỏi dataset.
- `manual_review/`: ảnh cần xem lại hoặc cần đổi nhãn.

## 4. Cách Upload Nhanh Nhất

### Cách Khuyến Nghị: Dùng Google Drive For Desktop

1. Cài Google Drive for desktop.
2. Tạo folder `traffic_vehicle_review` trong ổ Drive.
3. Copy toàn bộ nội dung từ:

```text
traffic_vehicle_classification/data/cleaned/
```

sang:

```text
traffic_vehicle_review/cleaned_for_review/
```

4. Chờ Google Drive sync xong.

Cách này thường ổn hơn kéo thả trên web vì dataset ảnh có rất nhiều file nhỏ.

### Cách Nhanh Trên Web

1. Mở Google Drive.
2. Tạo folder `traffic_vehicle_review`.
3. Vào folder đó, tạo `cleaned_for_review`, `rejected`, `manual_review`.
4. Kéo thả toàn bộ folder `data/cleaned` hoặc từng folder label vào `cleaned_for_review`.
5. Sau khi upload xong, đổi tên folder `cleaned` thành `cleaned_for_review` nếu cần.

Nếu upload bị chậm hoặc lỗi, nên upload từng class:

```text
bicycle/
boat/
bus/
car/
helicopter/
minibus/
motorcycle/
taxi/
train/
truck/
```

## 5. Cách Chia Việc Cho Nhóm

Chia theo class là dễ nhất:

```text
Người 1: bicycle, motorcycle
Người 2: car, taxi
Người 3: bus, minibus, truck
Người 4: boat, train, helicopter
```

Mỗi người chỉ sửa trong folder được giao để tránh xung đột.

## 6. Quy Tắc Lọc Ảnh

Giữ ảnh nếu:

- Phương tiện chính nhìn rõ.
- Ảnh thuộc đúng class của folder.
- Ảnh không quá mờ, không quá nhỏ.
- Ảnh không phải icon, logo, ảnh vẽ quá khác ảnh thực tế.

Chuyển sang `rejected/<label>/` nếu:

- Không có phương tiện.
- Sai class và không chắc nên chuyển sang class nào.
- Ảnh quá mờ, quá nhỏ hoặc bị che quá nhiều.
- Ảnh bị watermark nặng.
- Ảnh trùng hoặc gần trùng.
- Ảnh chỉ có đường ray, bến xe, biển báo, phong cảnh, nhưng không thấy phương tiện rõ.

Chuyển sang `manual_review/relabel_to_<label>/` nếu:

- Ảnh rõ phương tiện nhưng đang nằm sai class.
- Ví dụ ảnh taxi đang nằm trong `car`, thì chuyển sang `manual_review/relabel_to_taxi/`.
- Ví dụ ảnh minibus đang nằm trong `bus`, thì chuyển sang `manual_review/relabel_to_minibus/`.

Sau khi nhóm thống nhất, các ảnh trong `manual_review/relabel_to_<label>/` sẽ được đưa vào folder class tương ứng.

## 7. Lưu Ý Khi Review Trên Drive

- Bật chế độ grid view để xem ảnh nhanh hơn.
- Không đổi tên folder class.
- Không tạo class mới nếu chưa thống nhất với nhóm.
- Không xóa vĩnh viễn ngay nếu còn nghi ngờ; hãy chuyển sang `manual_review`.
- Nếu nhiều người cùng làm, nên mỗi người chỉ phụ trách một vài class.
- Có thể dùng Google Sheet để ghi tiến độ review.

Ví dụ sheet tiến độ:

```text
class,assigned_to,status,notes
bicycle,Người 1,done,
boat,Người 4,in_progress,
bus,Người 3,in_progress,cần xem kỹ minibus
car,Người 2,in_progress,cần tách taxi
```

## 8. Sau Khi Lọc Xong Thì Download Như Thế Nào?

Sau khi review xong, folder dùng để train là:

```text
traffic_vehicle_review/cleaned_for_review/
```

Trên Google Drive:

1. Click chuột phải vào folder `cleaned_for_review`.
2. Chọn `Download`.
3. Google Drive sẽ nén thành file `.zip`.
4. Tải file `.zip` về máy.
5. Giải nén và đặt lại vào project local.

Ví dụ đặt về:

```text
traffic_vehicle_classification/data/cleaned/
```

Nếu muốn giữ bản cũ, có thể đặt thành:

```text
traffic_vehicle_classification/data/cleaned_reviewed/
```

Khi train trên Kaggle, nên upload dataset đã review theo cấu trúc:

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

Notebook sẽ tự chia train/validation/test, nên không bắt buộc upload `data/splits`.

## 9. Flow Ngắn Gọn Nhất

```text
Local data/cleaned
  ↓
Upload lên Google Drive thành cleaned_for_review
  ↓
Mời mọi người vào Drive
  ↓
Mỗi người lọc một vài class
  ↓
Ảnh đúng: giữ nguyên
  ↓
Ảnh sai/xấu: chuyển sang rejected hoặc manual_review
  ↓
Sau khi lọc xong: download cleaned_for_review
  ↓
Đưa về local hoặc upload lên Kaggle
  ↓
Notebook tự split và train MobileNet/ResNet
```

## 10. Trả Lời Câu Hỏi Hay Gặp

### Có nên upload nguyên folder `cleaned` không?

Có. Với Google Drive, nên upload nguyên folder `cleaned` hoặc copy toàn bộ nội dung của nó vào `cleaned_for_review`.

### Có cần upload từng folder class không?

Không bắt buộc. Nhưng nếu mạng yếu hoặc upload bị lỗi, upload từng folder class sẽ dễ kiểm soát hơn.

### Có cần dùng `splits` không?

Chưa cần. `splits` chỉ dùng khi đã chốt dataset cuối cùng. Với Kaggle notebook hiện tại, chỉ cần upload:

```text
data/cleaned/<label>/*.jpg
```

Notebook sẽ tự stratified split.

### Nếu ảnh sai nhãn thì làm gì?

Nếu chắc chắn ảnh thuộc class khác, chuyển vào:

```text
manual_review/relabel_to_<class_dung>/
```

Nếu ảnh xấu hoặc không dùng được, chuyển vào:

```text
rejected/<class_hien_tai>/
```

### Sau review có cần chạy lại thống kê không?

Có. Sau khi tải dataset đã review về local, chạy lại:

```bash
python scripts/dataset_statistics.py --input data/cleaned
```

Sau đó mới upload lên Kaggle để train.

