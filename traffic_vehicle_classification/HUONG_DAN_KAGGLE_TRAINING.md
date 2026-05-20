# Hướng dẫn train trên Kaggle (bản cập nhật)

## 1) Chuẩn bị data local

Chạy pipeline trước khi zip:

```bash
python scripts/remove_corrupted_images.py --clear-output
python scripts/remove_duplicate_images.py --method phash --threshold 4 --action move
python scripts/balance_to_target.py --min-keep 1100 --max-keep 1200 --drop-labels taxi --action move
python scripts/dataset_statistics.py
```

## 2) Tạo file zip upload Kaggle

```bash
python scripts/create_kaggle_cleaned_zip.py --source data/cleaned --output traffic_vehicle_cleaned_kaggle.zip
```

## 3) Upload dataset lên Kaggle

Tạo Dataset mới và upload `traffic_vehicle_cleaned_kaggle.zip`.

## 4) Mở notebook chính

```text
notebooks/traffic_vehicle_classification_full_pipeline.ipynb
```

Notebook tự tìm dữ liệu trong Kaggle Input. Nếu cần, override `DATA_ROOT` ở cuối Section 2.

## 5) Cấu hình huấn luyện khuyến nghị cho T4x2

- `PER_REPLICA_BATCH_SIZE = 32`
- `EPOCHS = 35`
- `LEARNING_RATE = 1e-3`
- `USE_MIXUP = True`, `MIXUP_ALPHA = 0.15`
- `USE_FOCAL_LOSS = True`
- `USE_CLASS_WEIGHTS = False` (chỉ bật khi thật sự cần)

Mô hình:

- MobileNetV2 from scratch (`weights=None`)
- ResNet18 from scratch (custom)

## 6) Lưu ý để tránh lỗi

- Nếu vừa sửa kiến trúc model, hãy **Restart Session** rồi `Run All`.
- Nếu thấy training bất thường, kiểm tra lại:
  - `DATA_ROOT` trỏ đúng `cleaned/`
  - class count mỗi label không lệch quá mạnh
  - không còn ảnh hỏng trong `cleaned/`

## 7) Artifact sau train

Kaggle lưu ở:

```text
/kaggle/working/training_artifacts
```

Bao gồm:

- `models/*_best.keras`
- `models/*_final.keras`
- `*_history.csv`
- `*_classification_report.csv`
- `*_test_predictions.csv`
- `figures/*`
