# Huong Dan Chay Pipeline Crawl Va Xu Ly Data

File nay nam trong `data/` de ban mo nhanh khi bat dau thu thap du lieu. Tat ca lenh ben duoi chay tu thu muc project:

```bash
cd traffic_vehicle_classification
```

## 0. Cai Moi Truong

```bash
pip install -r requirements.txt
```

Neu chi muon chay cac script crawl/clean/statistics ma chua train notebook, co the tam bo qua `tensorflow` neu cai qua nang tren may local. Khi train tren Kaggle/Colab thi TensorFlow thuong da co san.

Lenh cai nhe cho giai doan crawl/clean local:

```bash
pip install pyyaml requests pillow tqdm imagehash pandas matplotlib seaborn scikit-learn ddgs selenium webdriver-manager
```

AutoCrawler dung Selenium nen may local can cai Google Chrome. Thu muc `autocrawler/` phai nam canh `traffic_vehicle_classification/`:

```text
vehicle-classification/
├── autocrawler/
└── traffic_vehicle_classification/
```

## 1. Kiem Tra Config Va Lenh Crawl

Chay dry-run de xem script doc label/keyword va goi AutoCrawler nhu the nao. Lenh nay khong tai anh:

```bash
python scripts/crawl_images.py --dry-run --labels car --limit-per-label 20
```

Neu thay command AutoCrawler hien ra dung, co the bat dau crawl that.

## 2. Crawl That Theo Tung Class

Nen crawl thu nho truoc de kiem tra Chrome/Selenium:

```bash
python scripts/crawl_images.py --labels car --limit-per-label 30
```

Neu thanh cong, tang so luong:

```bash
python scripts/crawl_images.py --labels car,bus,truck --limit-per-label 1500
```

Crawl tat ca 10 class:

```bash
python scripts/crawl_images.py --limit-per-label 1500
```

Anh raw duoc luu vao:

```text
data/raw/<label>/
```

Log nguon crawl, keyword va duong dan anh:

```text
reports/crawl_metadata.csv
```

Neu Google/Naver bi loi do Selenium hoac thay doi giao dien, co the chay DuckDuckGo fallback:

```bash
python scripts/crawl_images.py --sources duckduckgo --labels car --limit-per-label 300
```

## 3. Lam Sach Anh

Buoc nay mo tung anh, loai anh loi, loai anh qua nho, chuan hoa RGB JPEG va doi ten theo format `label_000001.jpg`.

```bash
python scripts/remove_corrupted_images.py
```

Ket qua:

```text
data/cleaned/<label>/
reports/cleaning_report.csv
```

## 4. Loc Anh Trung Lap

Dung pHash hoac dHash, so sanh Hamming distance. Neu distance nho hon hoac bang threshold thi coi la anh trung/gan trung.

```bash
python scripts/remove_duplicate_images.py --method phash --threshold 6 --action move
```

Anh trung duoc chuyen vao:

```text
data/duplicates/<label>/
reports/duplicates.csv
```

Neu chi muon so sanh trung trong tung label, them:

```bash
--within-label-only
```

## 5. Loc Thu Cong

Tao grid HTML de xem nhanh anh:

```bash
python scripts/manual_filter_helper.py --label car --max-per-label 120
```

Mo file:

```text
reports/manual_review.html
```

Sua `reports/manual_review_manifest.csv`:

- `keep`: giu anh
- `reject`: chuyen sang `data/rejected/<label>/`
- `review`: chuyen sang `data/manual_review/<label>/`
- `move`: chuyen sang label khac, can dien `target_label`

Ap dung thao tac:

```bash
python scripts/manual_filter_helper.py --apply-actions reports/manual_review_manifest.csv
```

## 6. Thong Ke Dataset

```bash
python scripts/dataset_statistics.py
```

Ket qua:

```text
reports/dataset_summary.csv
reports/image_details.csv
reports/image_errors.csv
reports/figures/
```

## 7. Chia Train / Validation / Test

Mac dinh chia stratified theo ty le 70/15/15:

```bash
python scripts/split_dataset.py --clear-output
```

Ket qua:

```text
data/splits/train/<label>/
data/splits/val/<label>/
data/splits/test/<label>/
reports/split_summary.csv
```

## 8. Chay Notebook Tong Hop

Notebook chinh:

```text
notebooks/traffic_vehicle_classification_full_pipeline.ipynb
```

Khi upload len Kaggle, hay upload folder dataset da co `data/cleaned/` hoac `data/splits/`. Trong Section 2 cua notebook, kiem tra bien:

```python
DATA_ROOT
RUN_TRAINING
MAX_IMAGES_PER_CLASS
EXPORT_SPLIT_FOLDERS
```

De test nhanh:

```python
MAX_IMAGES_PER_CLASS = 200
EPOCHS = 3
RUN_TRAINING = True
```

De train that:

```python
MAX_IMAGES_PER_CLASS = None
EPOCHS = 30
RUN_TRAINING = True
```

## Goi Y Quy Trinh Thuc Te

1. Crawl moi class khoang 1500 anh raw.
2. Chay clean de tao `data/cleaned/`.
3. Chay deduplicate.
4. Chay manual review cho class de nham, nhu `car`, `taxi`, `bus`, `minibus`, `truck`.
5. Chay statistics va xem bieu do phan bo class.
6. Neu class nao thieu, crawl bo sung class do.
7. Chia train/val/test.
8. Upload len Kaggle va chay notebook train MobileNet/ResNet from scratch.

## Lenh Tu Dong Lap Den Khi Du Anh Sach

Neu muon script tu lap `crawl -> clean -> deduplicate -> statistics -> check count`, dung:

```bash
python scripts/run_pipeline_until_ready.py --sources google,naver,duckduckgo --target-clean-per-label 1500 --batch-size 300
```

Giai thich:

- `--target-clean-per-label 1500`: moi label can dat 1500 anh sach trong `data/cleaned/<label>/`.
- `--batch-size 300`: moi vong crawl them toi da 300 anh raw cho moi label dang thieu.
- `--sources google,naver,duckduckgo`: dung AutoCrawler cho Google/Naver va DuckDuckGo fallback, phu hop muc tieu 1500 anh sach/label.
- Neu chi muon chay khong can Chrome, dung `--sources duckduckgo`, nhung co the khong du 1500 anh/label.
- Co the dung lai giua chung; chay lai lenh tren se resume dua tren so anh hien co.

Log tien do:

```text
reports/pipeline_progress.csv
```

Neu muon test nho:

```bash
python scripts/run_pipeline_until_ready.py --sources duckduckgo --labels car --target-clean-per-label 10 --batch-size 5 --max-rounds 2
```
