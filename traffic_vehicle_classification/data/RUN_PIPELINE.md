# RUN PIPELINE (Bản mới)

Flow hiện tại:

1. Crawl (`raw`)
2. Clean (`cleaned`)
3. Deduplicate (`duplicates`)
4. Balance về `1100-1200` ảnh / class + bỏ `taxi`
5. Statistics
6. Split `train/val/test`
7. Zip và upload Kaggle

## 1) Crawl

```bash
python scripts/crawl_images.py --limit-per-label 1200
```

## 2) Clean

```bash
python scripts/remove_corrupted_images.py --clear-output
```

## 3) Deduplicate

```bash
python scripts/remove_duplicate_images.py --method phash --threshold 4 --action move
```

## 4) Balance + Drop taxi

```bash
python scripts/balance_to_target.py --min-keep 1100 --max-keep 1200 --drop-labels taxi --action move
```

## 5) Statistics

```bash
python scripts/dataset_statistics.py
```

## 6) Split

```bash
python scripts/split_dataset.py --clear-output
```

## 7) Zip cleaned để upload Kaggle

```bash
python scripts/create_kaggle_cleaned_zip.py --source data/cleaned --output traffic_vehicle_cleaned_kaggle.zip
```
