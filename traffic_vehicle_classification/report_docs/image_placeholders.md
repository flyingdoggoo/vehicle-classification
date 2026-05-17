# Mapping Placeholder Hình Ảnh Cho Báo Cáo

File này dùng để quản lý toàn bộ hình cần thay vào `bao_cao_khdl.md`. Khi đã có ảnh thật, cập nhật cột `Nguồn ảnh thật` và thay đường dẫn trong báo cáo chính.

| Mã placeholder | Vị trí trong báo cáo | Mô tả hình cần thay | Nguồn ảnh thật dự kiến | Trạng thái |
|---|---|---|---|---|
| PLACEHOLDER_FIG_CRAWL_01 | 1.1 | Màn hình terminal/notebook đang chạy pipeline crawl dữ liệu | Screenshot khi chạy `run_pipeline_until_ready.py` | Chờ thay |
| PLACEHOLDER_FIG_CRAWL_02 | 2.2 | Cấu hình label và keyword trong `labels.yaml` | Screenshot file config hoặc bảng keyword | Chờ thay |
| PLACEHOLDER_FIG_CRAWL_03 | 2.4 | Log tiến độ crawl theo từng label | `reports/pipeline_progress.csv` hoặc screenshot terminal | Chờ thay |
| PLACEHOLDER_FIG_FLOW_01 | 1.5 | Flow tổng quan dự án | Render `mermaid/project_overall_flow.mmd` | Chờ thay |
| PLACEHOLDER_FIG_CLEAN_01 | 2.5 | Ví dụ ảnh lỗi, ảnh quá nhỏ, ảnh bị loại | Ảnh từ `data/rejected` hoặc screenshot notebook | Chờ thay |
| PLACEHOLDER_FIG_CLEAN_02 | 2.6 | Ví dụ ảnh trùng/gần trùng bằng pHash | Ảnh từ `data/duplicates` | Chờ thay |
| PLACEHOLDER_FIG_CLEAN_03 | 2.7 | Giao diện/grid lọc thủ công dữ liệu | `reports/manual_review.html` hoặc Google Drive screenshot | Chờ thay |
| PLACEHOLDER_FIG_DATA_01 | 2.8 | Biểu đồ phân bố số ảnh theo class | `reports/figures/class_distribution.png` hoặc notebook Section 4 | Chờ thay |
| PLACEHOLDER_FIG_PRETRAIN_01 | 3.1 | Ảnh mẫu theo từng class trước train | Notebook Section 4 sample grid | Chờ thay |
| PLACEHOLDER_FIG_PRETRAIN_02 | 3.2 | Histogram width/height của ảnh | Notebook Section 4 hoặc `reports/figures/*histogram.png` | Chờ thay |
| PLACEHOLDER_FIG_PRETRAIN_03 | 3.2 | Boxplot width/height/file size | Notebook Section 4 hoặc `reports/figures/*boxplot.png` | Chờ thay |
| PLACEHOLDER_FIG_PRETRAIN_04 | 3.3 | Ảnh trước và sau resize/normalize | Notebook Section 7 | Chờ thay |
| PLACEHOLDER_FIG_PRETRAIN_05 | 3.4 | Minh họa data augmentation trên train set | Notebook Section 7 | Chờ thay |
| PLACEHOLDER_FIG_PRETRAIN_06 | 3.5 | PCA hoặc t-SNE đặc trưng ảnh trước train | Notebook Section 8 | Chờ thay |
| PLACEHOLDER_FIG_SPLIT_01 | 3.6 | So sánh phân bố class train/val/test | Notebook Section 6 | Chờ thay |
| PLACEHOLDER_FIG_MODEL_01 | 4.3 | Kiến trúc tổng quan MobileNet | Sơ đồ tự vẽ hoặc screenshot `model.summary()` | Chờ thay |
| PLACEHOLDER_FIG_MODEL_02 | 4.4 | Kiến trúc tổng quan ResNet | Sơ đồ tự vẽ hoặc screenshot `model.summary()` | Chờ thay |
| PLACEHOLDER_FIG_TRAIN_01 | 4.6 | Training/validation loss và accuracy của MobileNet | `training_artifacts/figures/mobile...training_curves.png` | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_TRAIN_02 | 4.7 | Training/validation loss và accuracy của ResNet | `training_artifacts/figures/resnet...training_curves.png` | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_EVAL_01 | 4.8 | Confusion matrix MobileNet | `training_artifacts/figures/mobile...confusion_matrix.png` | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_EVAL_02 | 4.8 | Confusion matrix ResNet | `training_artifacts/figures/resnet...confusion_matrix.png` | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_EVAL_03 | 4.9 | Bar chart so sánh metric MobileNet và ResNet | `training_artifacts/figures/model_metric_comparison.png` | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_PRED_01 | 4.10 | 5 ảnh dự đoán đúng mỗi label của MobileNet | Notebook Section 11 visualization | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_PRED_02 | 4.10 | Ảnh dự đoán sai của MobileNet | Notebook Section 11 visualization | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_PRED_03 | 4.10 | 5 ảnh dự đoán đúng mỗi label của ResNet | Notebook Section 11 visualization | Chờ chạy Kaggle |
| PLACEHOLDER_FIG_PRED_04 | 4.10 | Ảnh dự đoán sai của ResNet | Notebook Section 11 visualization | Chờ chạy Kaggle |

## Placeholder bảng kết quả

| Mã placeholder | Ý nghĩa | Nguồn sau train |
|---|---|---|
| PLACEHOLDER_MOBILENET_BEST_VAL_ACC | Best validation accuracy MobileNet | `mobilenet_history.csv` |
| PLACEHOLDER_MOBILENET_TEST_ACC | Test accuracy MobileNet | `model_comparison.csv` |
| PLACEHOLDER_MOBILENET_PRECISION | Precision MobileNet | `model_comparison.csv` |
| PLACEHOLDER_MOBILENET_RECALL | Recall MobileNet | `model_comparison.csv` |
| PLACEHOLDER_MOBILENET_F1 | F1-score MobileNet | `model_comparison.csv` |
| PLACEHOLDER_MOBILENET_PARAMS | Số tham số MobileNet | `model_comparison.csv` |
| PLACEHOLDER_MOBILENET_TRAIN_TIME | Thời gian train MobileNet | `model_comparison.csv` |
| PLACEHOLDER_RESNET_BEST_VAL_ACC | Best validation accuracy ResNet | `resnet_history.csv` |
| PLACEHOLDER_RESNET_TEST_ACC | Test accuracy ResNet | `model_comparison.csv` |
| PLACEHOLDER_RESNET_PRECISION | Precision ResNet | `model_comparison.csv` |
| PLACEHOLDER_RESNET_RECALL | Recall ResNet | `model_comparison.csv` |
| PLACEHOLDER_RESNET_F1 | F1-score ResNet | `model_comparison.csv` |
| PLACEHOLDER_RESNET_PARAMS | Số tham số ResNet | `model_comparison.csv` |
| PLACEHOLDER_RESNET_TRAIN_TIME | Thời gian train ResNet | `model_comparison.csv` |
| PLACEHOLDER_COMMENT_LEVEL | Nhận xét định tính về mức kết quả MobileNet | Tự viết sau khi xem metrics và prediction samples |
| PLACEHOLDER_COMPARE_PARAMS | Nhận xét so sánh số tham số ResNet với MobileNet | `model_comparison.csv` |
| PLACEHOLDER_COMPARE_TIME | Nhận xét so sánh thời gian train ResNet với MobileNet | `model_comparison.csv` |
| PLACEHOLDER_BEST_MODEL | Tên mô hình tốt hơn theo macro F1-score | `model_comparison.csv` |
