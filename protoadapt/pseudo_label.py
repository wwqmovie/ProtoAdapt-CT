
"""ProtoAdapt-CT prototype construction and pseudo-label assignment (S1-S3)."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

def build_prototypes(train_csv, feature_dir=None):
    """Build class prototypes from labeled training features.

    Args:
        train_csv: path to CSV with feature_path, label columns
    Returns:
        proto_0, proto_1: class-average feature vectors (torch.Tensor, 400-dim)
        stats: dict with Q1/Q3 per class for 3-tier thresholds
    """
    df = pd.read_csv(train_csv)
    feats_0, feats_1 = [], []
    for _, r in df.iterrows():
        fp = r["feature_path"]
        feat = torch.load(fp, map_location="cpu", weights_only=False).float()
        if int(r["label"]) == 0:
            feats_0.append(feat)
        else:
            feats_1.append(feat)

    proto_0 = torch.stack(feats_0).mean(0)
    proto_1 = torch.stack(feats_1).mean(0)

    # Compute per-class similarity statistics
    sims_0 = torch.stack([F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0))
                          for f in feats_0]).numpy()
    sims_1 = torch.stack([F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0))
                          for f in feats_1]).numpy()

    stats = {
        "cos_sim": F.cosine_similarity(proto_0.unsqueeze(0), proto_1.unsqueeze(0)).item(),
        "class0_q1": float(np.percentile(sims_0, 25)), "class0_q3": float(np.percentile(sims_0, 75)),
        "class1_q1": float(np.percentile(sims_1, 25)), "class1_q3": float(np.percentile(sims_1, 75)),
        "class0_mean": float(sims_0.mean()), "class1_mean": float(sims_1.mean()),
    }
    return proto_0, proto_1, stats


def assign_3tier_pseudo(pretrain_csv, proto_0, proto_1, stats):
    """Assign 3-tier pseudo-labels to pretrain features.

    High confidence (sim > Q3): directly assign label.
    Medium (Q1 < sim < Q3): assign with anchor classifier (placeholder).
    Low (sim < Q1): discard.

    Args:
        pretrain_csv: CSV with feature_path, similarity_to_class_0_average,
                      similarity_to_class_1_average columns
        proto_0, proto_1: class prototypes
        stats: dict from build_prototypes()

    Returns:
        df_pseudo: DataFrame with feature_path, label, need_pull, sim_to_pseudo
    """
    df = pd.read_csv(pretrain_csv)
    q1_0, q3_0 = stats["class0_q1"], stats["class0_q3"]
    q1_1, q3_1 = stats["class1_q1"], stats["class1_q3"]

    results = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="3-tier assign"):
        s0 = r["similarity_to_class_0_average"]
        s1 = r["similarity_to_class_1_average"]
        ms = max(s0, s1)
        lbl = 0 if s0 > s1 else 1
        q3 = q3_0 if lbl == 0 else q3_1
        q1 = q1_0 if lbl == 0 else q1_1

        if ms >= q3:
            results.append({"feature_path": r["feature_path"], "label": lbl,
                           "need_pull": True, "sim_to_pseudo": ms})
        elif ms >= q1:
            # Medium confidence: keep but mark for anchor classifier
            results.append({"feature_path": r["feature_path"], "label": lbl,
                           "need_pull": False, "sim_to_pseudo": ms})
        # Low confidence: discard

    return pd.DataFrame(results)


def top20_pseudo(pretrain_feat_dir, proto_0, proto_1, n_sample=100000, n_tau=20000):
    """Assign Top20% pseudo-labels (alternative for high proto cos_sim tasks).

    Args:
        pretrain_feat_dir: directory of pretrain .pt feature files
        proto_0, proto_1: class prototypes
        n_sample: number of pretrain samples to assign pseudo-labels
        n_tau: number of samples for tau estimation

    Returns:
        df_pseudo, tau, tau_stats
    """
    import random, os
    random.seed(42)
    all_pt = sorted([f for f in os.listdir(pretrain_feat_dir) if f.endswith(".pt")])

    # Compute tau
    spl = random.sample(all_pt, min(n_tau, len(all_pt)))
    s0s, s1s = [], []
    for fn in spl:
        f = torch.load(os.path.join(pretrain_feat_dir, fn), map_location="cpu",
                       weights_only=False).float()
        s0s.append(F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item())
        s1s.append(F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item())
    s0s, s1s = np.array(s0s), np.array(s1s)
    tau = np.percentile(np.maximum(s0s, s1s), 80)

    # Assign pseudo-labels
    use = random.sample(all_pt, min(n_sample, len(all_pt)))
    rows = []
    for fn in tqdm(use, desc="Top20% assign"):
        fp = os.path.join(pretrain_feat_dir, fn)
        f = torch.load(fp, map_location="cpu", weights_only=False).float()
        s0 = F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item()
        s1 = F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item()
        if max(s0, s1) > tau:
            rows.append({"feature_path": fp, "label": 0 if s0 > s1 else 1})

    tau_stats = {"tau": tau, "s0_p30": float(np.percentile(s0s, 30)),
                 "s0_p70": float(np.percentile(s0s, 70)),
                 "s1_p30": float(np.percentile(s1s, 30)),
                 "s1_p70": float(np.percentile(s1s, 70))}
    return pd.DataFrame(rows), tau_stats
