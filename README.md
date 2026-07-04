# ProtoAdapt-CT

Report-Consistent Prototype Learning for Lung Cancer Assessment via CT Foundation Model Adaptation.

## Quick Start

### 1. Organize data

```
data/
├── pretrain/
│   ├── ct/                    # upstream CTs (.nii.gz)
│   ├── ct_list.csv            # column: img_path
│   ├── reports/               # radiology reports (.txt)
│   └── features/              # (generated)
└── downstream/
    ├── ct/                    # labeled CTs (.nii.gz)
    ├── ct_list.csv            # column: img_path
    ├── features/              # (generated)
    └── {task}/                # one subdirectory per task
        ├── train.csv          # columns: feature_path, label
        └── test.csv           # columns: feature_path, label
```

### 2. Extract features

```bash
python scripts/extract_features.py \
    --input_csv data/pretrain/ct_list.csv \
    --output_dir data/pretrain/features/ \
    --pretrained /path/to/swin3d_checkpoint.safetensors

python scripts/extract_features.py \
    --input_csv data/downstream/ct_list.csv \
    --output_dir data/downstream/features/ \
    --pretrained /path/to/swin3d_checkpoint.safetensors
```

### 3. Run

```bash
python main.py --task lung1 --pretrain_dir data/pretrain/features/

# With S4 report filtering:
python main.py --task lung1 --pretrain_dir data/pretrain/features/ \
    --ct_mapping data/ct_report_mapping.csv \
    --lamed_path /path/to/lammed
```

Add a new task by creating `data/downstream/{task}/train.csv` and `test.csv`.

## Pipeline

| Stage | Description |
|-------|-------------|
| S1-S3 (Image) | Build prototypes from labeled features, assign pseudo-labels to pretrain features via Top20% threshold |
| S4 (Text) | LaMed generates English reports from CT, SBERT compares with original reports, filters inconsistent pseudo-labels |
| FP | Feature calibration pulls pseudo-label features toward class prototypes |
| Train | ResNet1D trains on labeled + calibrated pseudo-label data |

Output: `output/predictions.csv` and `output/best_model.pth`.

## Requirements

```
torch>=2.0  numpy  pandas  scikit-learn
datasets>=2.0  timm  tqdm
SimpleITK  opencv-python
sentence-transformers  transformers
```

## Project Structure

```
main.py                             # entry point
protoadapt/
├── models/                         # ResNet1D, ResNet1D_Easy, FC, Swin3D
├── pseudo_label.py                 # prototype construction + pseudo-labels
├── data_loader.py / _fp.py         # dataset loading (standard / feature-pull)
├── text/report_similarity.py       # S4: LaMed + SBERT
├── engine.py / utils.py / optim.py
scripts/
├── extract_features.py             # Swin3D feature extraction
├── assign_pseudo_labels.py         # standalone pseudo-labeling
└── s4_filter.py                    # standalone S4 filter
```

## License

MIT
