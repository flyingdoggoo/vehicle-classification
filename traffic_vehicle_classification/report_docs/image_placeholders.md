# Mapping Hình Ảnh Và Số Liệu Cho Báo Cáo

File này ghi lại các hình và sơ đồ đã được thay vào `bao_cao_khdl.md`. Các hình thuộc phần train/evaluate được trích xuất trực tiếp từ output của hai notebook `mobilenet.ipynb` và `resnet50.ipynb`. Các hình thuộc phần crawl/làm sạch được tạo từ log, CSV và thư mục dữ liệu hiện có trong project.

## Ghi chú mô hình

Hai notebook dùng `MODEL_NAME` lần lượt là `MobileNetV2` và `ResNet50V2`, backbone `trainable=True`, `weights=None`.

| Mã hình | Vị trí trong báo cáo | File đã dùng | Nguồn |
|---|---|---|---|
| FIG-CRAWL-01 | 1.1 | `figures/crawl_log_excerpt.png` | `reports/pipeline_run.log` |
| FIG-FLOW-01 | 1.4 | Mermaid embedded trong báo cáo | Tóm tắt flow hiện tại, đồng bộ `mermaid/project_overall_flow.mmd` |
| FIG-CRAWL-02 | 2.2 | `figures/label_keyword_config_summary.png` | `configs/labels.yaml` |
| FIG-CRAWL-03 | 2.4 | `figures/crawl_progress_summary.png` | `reports/pipeline_progress.csv` |
| FIG-CLEAN-01 | 2.5 | `figures/rejected_examples.png` | `data/rejected` |
| FIG-CLEAN-02 | 2.6 | `figures/duplicate_pairs.png` | `reports/duplicates.csv`, `data/duplicates`, `data/cleaned` |
| FIG-CLEAN-03 | 2.7 | `figures/manual_review_grid_preview.png` | `data/cleaned`, quy trình `manual_filter_helper.py` |
| FIG-DATA-01 | 2.8 | `figures/class_distribution.png` | `reports/figures/class_distribution.png` |
| FIG-PRETRAIN-02 | 3.2 | `figures/width_histogram.png` | `reports/figures/width_histogram.png` |
| FIG-PRETRAIN-03 | 3.2 | `figures/file_size_boxplot.png` | `reports/figures/file_size_boxplot.png` |
| FIG-PRETRAIN-04 | 3.3 | `figures/mobilenet_preprocess_augmentation.png` | `mobilenet.ipynb`, Section Feature Visualization |
| FIG-PRETRAIN-05 | 3.4 | `figures/resnet_preprocess_augmentation.png` | `resnet50.ipynb`, Section Feature Visualization |
| FIG-PRETRAIN-06 | 3.5 | `figures/mobilenet_tsne_pixel_features.png` | `mobilenet.ipynb`, t-SNE visualization |
| FIG-SPLIT-01 | 3.6 | `figures/dataset_distribution_mobilenet.png` | `mobilenet.ipynb`, split distribution |
| FIG-MODEL-01 | 4.3 | Mermaid embedded trong báo cáo, đồng bộ `mermaid/mobilenetv2_architecture_flow.mmd` | Cấu hình model trong `mobilenet.ipynb` |
| FIG-MODEL-02 | 4.4 | Mermaid embedded trong báo cáo, đồng bộ `mermaid/resnet50v2_architecture_flow.mmd` | Cấu hình model trong `resnet50.ipynb` |
| FIG-TRAIN-01 | 4.6 | `figures/mobilenet_training_curves.png` | `mobilenet.ipynb`, training curves |
| FIG-TRAIN-02 | 4.7 | `figures/resnet_training_curves.png` | `resnet50.ipynb`, training curves |
| FIG-EVAL-01 | 4.8 | `figures/mobilenet_confusion_matrix.png` | `mobilenet.ipynb`, confusion matrix |
| FIG-EVAL-02 | 4.8 | `figures/resnet_confusion_matrix.png` | `resnet50.ipynb`, confusion matrix |
| FIG-EVAL-03 | 4.9 | `figures/model_metric_comparison.png` | Tạo từ metric test của hai notebook |
| FIG-PRED-01 | 4.10 | `figures/mobilenet_correct_examples.png` | `mobilenet.ipynb`, correct examples |
| FIG-PRED-02 | 4.10 | `figures/mobilenet_wrong_predictions.png` | `mobilenet.ipynb`, wrong predictions |
| FIG-PRED-03 | 4.10 | `figures/resnet_correct_examples.png` | `resnet50.ipynb`, correct examples |
| FIG-PRED-04 | 4.10 | `figures/resnet_wrong_predictions.png` | `resnet50.ipynb`, wrong predictions |

## Số Liệu Đã Thay Vào Báo Cáo

| Metric | MobileNetV2 | ResNet50V2 |
|---|---:|---:|
| Total images | 10,767 | 10,767 |
| Train / Val / Test | 7,536 / 1,615 / 1,616 | 7,536 / 1,615 / 1,616 |
| Best validation accuracy | 0.9238, epoch 32 | 0.9300, epoch 21 |
| Test accuracy | 0.9152 | 0.9208 |
| Macro precision | 0.9164 | 0.9238 |
| Macro recall | 0.9154 | 0.9209 |
| Macro F1-score | 0.9153 | 0.9217 |
| Parameters | 2,269,513 | 23,583,241 |
| Training time | 1,177.9 s | 1,567.1 s |
| Epoch đã chạy | 44 | 33 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-5 | 1e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Loss | Sparse CE + label smoothing 0.05 | Sparse CE + label smoothing 0.05 |
