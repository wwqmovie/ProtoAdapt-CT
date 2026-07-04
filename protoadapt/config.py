"""All configurable paths and hyperparameters. Fill in before running."""

import os

# ---------- data paths (user must configure) ----------
DATA = {
    "huaxi": {
        "train": "data/huaxi/train.csv",
        "val": "data/huaxi/val.csv",
        "s3_pseudo": "data/huaxi/pseudo_label_results.csv",
        "s4_full": "data/huaxi/report_filtered_full.csv",
        "proto_0": "data/huaxi/class_0_average.pt",
        "proto_1": "data/huaxi/class_1_average.pt",
    },
    "lung1": {
        "train": "data/lung1/train.csv",
        "test": "data/lung1/test.csv",
        "val": "data/lung1/val.csv",
        "s3_pseudo": "data/lung1/s3_pseudo.csv",
        "s4_full": "data/lung1/s4_full.csv",
        "proto_0": "data/lung1/proto_0.pt",
        "proto_1": "data/lung1/proto_1.pt",
    },
    "egfr": {
        "train": "data/egfr/train.csv",
        "test": "data/egfr/test.csv",
        "proto_0": "data/egfr/class_0_average.pt",
        "proto_1": "data/egfr/class_1_average.pt",
    },
    "n": {
        "train": "data/n/train.csv",
        "test": "data/n/test.csv",
        "proto_0": "data/n/class_0_average.pt",
        "proto_1": "data/n/class_1_average.pt",
    },
    "m": {
        "train": "data/m/train.csv",
        "test": "data/m/test.csv",
        "proto_0": "data/m/class_0_average.pt",
        "proto_1": "data/m/class_1_average.pt",
    },
}

# External test datasets (for EGFR/N/M)
EXTERNAL = {
    "EGFR_Prosp": "data/external/egfr_prosp.csv",
    "EGFR_TCIA": "data/external/egfr_tcia.csv",
    "N_Shengjing": "data/external/n_shengjing.csv",
    "N_Jilin": "data/external/n_jilin.csv",
    "M_Shengjing": "data/external/m_shengjing.csv",
    "M_Jilin": "data/external/m_jilin.csv",
}

# Pretrain feature directories
PRETRAIN_FEAT_DIR = "data/pretrain_features/"
EXTERNAL_FEAT_DIR = "data/external_features/"

# S4 scores CSV (precomputed LaMed+SBERT report similarities)
S4_SCORES_CSV = "data/s4_report_scores.csv"

# ---------- training hyperparameters ----------
HP = {
    "batch_size": 256,
    "epochs": 200,
    "lr": 4e-3,
    "min_lr": 1e-8,
    "warmup_epochs": 20,
    "weight_decay": 0.05,
    "seed": 42,
    "num_workers": 0,
    "pin_mem": False,
    "use_amp": False,
    "pull_strength": 0.5,  # HUAXI: 0.5, LUNG1: 0.4
}

# S4 thresholds (best per task from paper)
S4_THRESHOLDS = {"HUAXI": 0.60, "LUNG1": 0.55, "EGFR": 0.60, "N": 0.65, "M": 0.55}

# ---------- output ----------
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
