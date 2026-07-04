"""
Dataset loading with on-the-fly feature pull (FP) calibration.

During data preprocessing, pseudo-label features are pulled toward their
assigned class prototype: f' = (1-alpha)*f + alpha*P_c.

The pull strength alpha is computed from the feature's similarity to the
prototype and class-specific Q1/Q3 thresholds.

Usage:
    from protoadapt.data_loader_fp import prepare_dataset_fp
    dl_train, dl_val, n_train, n_val = prepare_dataset_fp(args, num_tasks, global_rank,
                                                           proto_0, proto_1, stats0, stats1)
"""

import torch
import numpy as np
import os
from datasets import Features, Value, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader


def feature_pull(feature, target_feature, similarity, threshold_high, threshold_low,
                 pull_strength=0.5):
    """Calibrate feature toward class prototype.

    f' = (1-alpha)*f + alpha*P_c
    alpha = clip(pull_strength * (th_high - sim) / (th_high - th_low), 0, 0.5)

    Args:
        feature: current feature tensor (400-dim)
        target_feature: class prototype tensor (400-dim)
        similarity: cosine similarity between feature and prototype
        threshold_high: Q3 of training data similarities (upper bound)
        threshold_low: Q1 of training data similarities (lower bound)
        pull_strength: pull coefficient (0.5 for HUAXI, 0.4 for LUNG1)

    Returns:
        calibrated feature tensor
    """
    if threshold_high == threshold_low:
        return feature
    pull_factor = pull_strength * (threshold_high - similarity) / \
                  (threshold_high - threshold_low)
    pull_factor = np.clip(pull_factor, 0, 0.5)
    return (1 - pull_factor) * feature + pull_factor * target_feature


def prepare_dataset_fp(args, num_tasks, global_rank,
                       proto_0, proto_1, stats0, stats1):
    """Prepare dataset with on-the-fly feature pull.

    Reads CSV files (train_path, val_path, optional pre_path for pseudo-labels).
    Applies feature_pull during the .with_transform(preprocess) stage.

    CSV columns required:
        feature_path, label, need_pull (bool), sim_to_pseudo (float)

    Args:
        args: namespace with train_path, val_path, pre_path, column_names,
              labels_name, batch_size, num_workers, pin_mem, seed, pull_strength
        num_tasks: number of distributed tasks
        global_rank: global rank for distributed training
        proto_0: class 0 prototype tensor
        proto_1: class 1 prototype tensor
        stats0: dict with 'q1', 'q3' for class 0
        stats1: dict with 'q1', 'q3' for class 1

    Returns:
        data_loader_train, data_loader_val, len_train, len_val
    """

    my_features = Features({
        "feature_path": Value("string"),
        "label": Value("int64"),
        "need_pull": Value("bool"),
        "sim_to_pseudo": Value("float64"),
    })

    data_files = {"train": args.train_path, "val": args.val_path}
    if hasattr(args, "pre_path") and args.pre_path and os.path.exists(args.pre_path):
        data_files["pre_path"] = args.pre_path

    dataset = load_dataset("csv", data_files=data_files, features=my_features)

    feature_center = {0: proto_0, 1: proto_1}
    threshold = {
        0: (stats0["q3"], stats0["q1"]),
        1: (stats1["q3"], stats1["q1"]),
    }

    pull_strength = getattr(args, "pull_strength", 0.5)

    def preprocess(examples):
        n_samples = len(examples[args.column_names])
        normalized_features = []

        for i in range(n_samples):
            path = examples[args.column_names][i]
            need_pull = examples["need_pull"][i]
            label = examples["label"][i]
            sim = examples["sim_to_pseudo"][i]

            feature = torch.load(path, weights_only=True)

            if need_pull and label in feature_center:
                th_high, th_low = threshold[label]
                feature = feature_pull(feature, feature_center[label], sim,
                                       th_high, th_low, pull_strength)

            normalized = (feature - 0.5) / 0.5
            normalized_features.append(normalized)

        examples["features"] = normalized_features
        return examples

    train_dataset = dataset["train"].with_transform(preprocess)
    if "pre_path" in dataset:
        pre_dataset = dataset["pre_path"].with_transform(preprocess)
        train_dataset = concatenate_datasets([train_dataset, pre_dataset])

    val_dataset = dataset["val"].with_transform(preprocess)

    def collate_fn(examples):
        features = torch.stack([e["features"] for e in examples])
        features = features.to(memory_format=torch.contiguous_format).float()
        labels = torch.tensor([e[args.labels_name] for e in examples])
        return {"features": features, "labels": labels}

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    data_loader_train = DataLoader(
        train_dataset, sampler=sampler_train, collate_fn=collate_fn,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )
    data_loader_val = DataLoader(
        val_dataset, sampler=sampler_val, collate_fn=collate_fn,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    return data_loader_train, data_loader_val, len(train_dataset), len(val_dataset)
