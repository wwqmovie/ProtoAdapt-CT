"""
Component ablation for EGFR/N/M (Top20% pseudo-labeling).

These tasks have high prototype cosine similarity (>0.98), making the paper's
3-tier Q1/Q3 method infeasible. Top20% threshold on pretrain max-sim is used instead.

Usage:
    python -m protoadapt.ablation_top20 --task EGFR --strategy Image+FP+Text --dataset internal --gpu 0
"""

import argparse
import os
import random
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from protoadapt.models import ResNet1D
from protoadapt.data_loader import prepare_dataset
from protoadapt.engine import train_one_epoch, evaluate
from protoadapt.optim import create_optimizer
from protoadapt.utils import NativeScalerWithGradNormCount as NativeScaler, cosine_scheduler
from protoadapt.config import DATA, EXTERNAL, HP, S4_THRESHOLDS, S4_SCORES_CSV, PRETRAIN_FEAT_DIR, EXTERNAL_FEAT_DIR, OUTPUT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# External dataset metadata
EXT_META = {
    "Prosp": {"task": "EGFR", "csv": EXTERNAL["EGFR_Prosp"], "feat_dir": "EGFR-Prosp", "label_col": "label"},
    "TCIA": {"task": "EGFR", "csv": EXTERNAL["EGFR_TCIA"], "feat_dir": "EGFR-TCIA", "label_col": "label"},
    "Shengjing": None,  # resolved by task
    "Jilin": None,
}


def parse_args():
    p = argparse.ArgumentParser(description="ProtoAdapt-CT Top20% ablation")
    p.add_argument("--task", required=True, choices=["EGFR", "N", "M"])
    p.add_argument("--strategy", required=True,
                   choices=["Baseline", "Image", "Image+FP", "Image+Text",
                            "Image+FP+Text", "Image+Text+FP"])
    p.add_argument("--dataset", default="internal",
                   help="internal, Prosp, TCIA, Shengjing, Jilin")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def make_args(tr, val):
    class A: pass
    a = A()
    a.train_path = tr; a.val_path = val
    a.column_names = "feature_path"; a.labels_name = "label"
    a.batch_size = HP["batch_size"]; a.num_workers = HP["num_workers"]
    a.pin_mem = HP["pin_mem"]; a.seed = HP["seed"]
    a.epochs = HP["epochs"]; a.lr = HP["lr"]; a.min_lr = HP["min_lr"]
    a.warmup_epochs = HP["warmup_epochs"]; a.warmup_steps = -1
    a.weight_decay = HP["weight_decay"]; a.weight_decay_end = None
    a.update_freq = 1; a.clip_grad = None; a.use_amp = HP["use_amp"]
    a.distributed = False; a.save_ckpt = False; a.log_dir = None
    a.start_epoch = 0; a.device = DEVICE
    a.opt = "adamw"; a.opt_eps = 1e-8; a.opt_betas = None
    a.momentum = 0.9; a.layer_decay = 1.0; a.output_dir = "/tmp"
    a.pre_path = ""; a.pull_strength = HP["pull_strength"]
    return a


def build_dtotal_raw(df_train, df_ps):
    rows = [{"feature_path": r["feature_path"], "label": r["label"]}
            for _, r in df_train.iterrows()]
    for _, r in df_ps.iterrows():
        if os.path.exists(r["feature_path"]):
            rows.append({"feature_path": r["feature_path"], "label": int(r["label"])})
    return pd.DataFrame(rows)


def build_dtotal_calib(df_train, df_ps, proto_0, proto_1,
                       s0p30, s0p70, s1p30, s1p70, calib_dir):
    os.makedirs(calib_dir, exist_ok=True)
    rows = [{"feature_path": r["feature_path"], "label": r["label"]}
            for _, r in df_train.iterrows()]
    for _, r in df_ps.iterrows():
        fp = r["feature_path"]
        if not os.path.exists(fp): continue
        feat = torch.load(fp, map_location="cpu", weights_only=False).float()
        lbl = int(r["label"]); proto = proto_0 if lbl == 0 else proto_1
        s = F.cosine_similarity(feat.unsqueeze(0), proto.unsqueeze(0)).item()
        th = s0p70 if lbl == 0 else s1p70; tl = s0p30 if lbl == 0 else s1p30
        alpha = max(0.0, min(0.5, 0.5 * (th - s) / (th - tl + 1e-8)))
        if alpha > 0: feat = (1 - alpha) * feat + alpha * proto
        sp = os.path.join(calib_dir, os.path.basename(fp)); torch.save(feat, sp)
        rows.append({"feature_path": sp, "label": lbl})
    return pd.DataFrame(rows)


def train_eval(df_dt, test_csv, label_str, out_dir):
    tmp_tr = f"/tmp/{label_str}_tr.csv"; tmp_te = f"/tmp/{label_str}_te.csv"
    df_dt.to_csv(tmp_tr, index=False); pd.read_csv(test_csv).to_csv(tmp_te, index=False)
    args = make_args(tmp_tr, tmp_te)
    dl_tr, dl_val, n_tr, _ = prepare_dataset(args, num_tasks=1, global_rank=0)
    model = ResNet1D(input_dim=400, num_classes=2).to(DEVICE)
    opt = create_optimizer(args, model); crit = torch.nn.CrossEntropyLoss(); scaler = NativeScaler()
    spe = max(1, n_tr // HP["batch_size"])
    lr_s = cosine_scheduler(HP["lr"], HP["min_lr"], HP["epochs"], spe,
                            warmup_epochs=HP["warmup_epochs"])
    wd_s = cosine_scheduler(HP["weight_decay"], HP["weight_decay"] or HP["weight_decay"],
                            HP["epochs"], spe)

    max_auc, best_state = 0.0, None
    for ep in range(HP["epochs"]):
        train_one_epoch(model, crit, dl_tr, opt, DEVICE, ep, scaler, None, None, None,
                        start_steps=ep * spe, lr_schedule_values=lr_s, wd_schedule_values=wd_s,
                        num_training_steps_per_epoch=spe, update_freq=1, use_amp=False)
        ts = evaluate(dl_val, model, DEVICE)
        if ts["auc"] > max_auc:
            max_auc = ts["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "auc": max_auc},
                   os.path.join(out_dir, f"{label_str}_best.pth"))

    model.eval(); ap, al = [], []
    with torch.no_grad():
        for b in dl_val:
            o = model(b["features"].to(DEVICE))
            ap.extend(F.softmax(o, dim=1)[:, 1].cpu().numpy()); al.extend(b["labels"].numpy())
    ap, al = np.array(ap), np.array(al)
    df_te = pd.read_csv(test_csv)
    n = min(len(df_te), len(al))
    df_pred = pd.DataFrame({
        "feature_path": df_te["feature_path"].values[:n], "label": al[:n].astype(int),
        "prob_class0": 1 - ap[:n], "prob_class1": ap[:n], "pred": (ap[:n] >= 0.5).astype(int),
    })
    df_pred.to_csv(os.path.join(out_dir, f"{label_str}.csv"), index=False)
    print(f"  [{label_str}] AUC={max_auc:.4f}")
    return max_auc


def load_top20_pseudo(proto_0, proto_1, tau, n_files=100000):
    """Assign Top20% pseudo-labels from pretrain features."""
    all_pt = sorted([f for f in os.listdir(PRETRAIN_FEAT_DIR) if f.endswith(".pt")])
    use = random.sample(all_pt, min(n_files, len(all_pt)))
    rows = []
    for fn in tqdm(use, desc="  Top20% assign"):
        fp = os.path.join(PRETRAIN_FEAT_DIR, fn)
        f = torch.load(fp, map_location="cpu", weights_only=False).float()
        s0 = F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item()
        s1 = F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item()
        if max(s0, s1) > tau:
            rows.append({"feature_path": fp, "label": 0 if s0 > s1 else 1})
    return pd.DataFrame(rows)


def compute_tau_from_pretrain(proto_0, proto_1):
    """Compute Top20% tau from pretrain features."""
    all_pt = sorted([f for f in os.listdir(PRETRAIN_FEAT_DIR) if f.endswith(".pt")])
    spl = random.sample(all_pt, min(20000, len(all_pt)))
    s0s, s1s = [], []
    for fn in tqdm(spl, desc="  Sampling sims"):
        f = torch.load(os.path.join(PRETRAIN_FEAT_DIR, fn),
                       map_location="cpu", weights_only=False).float()
        s0s.append(F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item())
        s1s.append(F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item())
    s0s, s1s = np.array(s0s), np.array(s1s)
    tau = np.percentile(np.maximum(s0s, s1s), 80)
    return tau, float(np.percentile(s0s, 30)), float(np.percentile(s0s, 70)), \
           float(np.percentile(s1s, 30)), float(np.percentile(s1s, 70))


def load_external_test(dataset_name, task):
    """Load external test features."""
    ext_cfg = EXT_META[dataset_name]
    if ext_cfg is None:
        if task == "N":
            ext_cfg = {"csv": EXTERNAL[f"N_{dataset_name}"], "feat_dir": dataset_name, "label_col": "label_n_01"}
        elif task == "M":
            ext_cfg = {"csv": EXTERNAL[f"M_{dataset_name}"], "feat_dir": dataset_name, "label_col": "label_m"}
    df = pd.read_csv(ext_cfg["csv"])
    lc = ext_cfg["label_col"]
    df = df[df[lc].notna()].copy(); df[lc] = df[lc].astype(int)
    feat_dir = os.path.join(EXTERNAL_FEAT_DIR, ext_cfg["feat_dir"])
    rows = []
    for _, r in df.iterrows():
        nii = str(r["lungmulmask"])
        md5 = hashlib.md5(nii.encode()).hexdigest() + ".pt"
        fp = os.path.join(feat_dir, md5)
        if os.path.exists(fp):
            rows.append({"feature_path": fp, "label": int(r[lc])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    th = S4_THRESHOLDS[cfg.task]
    tag = f"{cfg.task}_{cfg.dataset}_{cfg.strategy.replace('+','_')}"
    out_dir = os.path.join(OUTPUT_DIR, "ablation_top20",
                           cfg.dataset if cfg.dataset != "internal" else "internal", tag)
    os.makedirs(out_dir, exist_ok=True)
    random.seed(HP["seed"]); np.random.seed(HP["seed"])
    print(f"Task={cfg.task} Strategy={cfg.strategy} Dataset={cfg.dataset} S4_th={th}")

    # Load labeled data
    dcfg = DATA[cfg.task.lower()]
    df_tr = pd.read_csv(dcfg["train"])
    df_train = df_tr[["feature_path", "label"]]

    if cfg.dataset == "internal":
        df_test = pd.read_csv(dcfg["test"])[["feature_path", "label"]]
    else:
        df_test = load_external_test(cfg.dataset, cfg.task)

    test_csv = f"/tmp/test_{tag}.csv"; df_test.to_csv(test_csv, index=False)
    print(f"  Train={len(df_train)} Test={len(df_test)}")

    # Prototypes
    proto_0 = torch.load(dcfg["proto_0"], map_location="cpu", weights_only=False).float()
    proto_1 = torch.load(dcfg["proto_1"], map_location="cpu", weights_only=False).float()
    pcos = F.cosine_similarity(proto_0.unsqueeze(0), proto_1.unsqueeze(0)).item()
    print(f"  Proto cos_sim: {pcos:.4f}")

    # Tau and calibration stats from pretrain
    tau, sp30_0, sp70_0, sp30_1, sp70_1 = compute_tau_from_pretrain(proto_0, proto_1)
    print(f"  Tau (Top20%): {tau:.4f}")

    # Pseudo-labels
    df_pseudo = load_top20_pseudo(proto_0, proto_1, tau)
    print(f"  Top20% pseudo: {len(df_pseudo)}")

    # S4 scores
    s4_map = {}
    if os.path.exists(S4_SCORES_CSV):
        df_s4 = pd.read_csv(S4_SCORES_CSV)
        s4_map = dict(zip(df_s4["feature_path"], df_s4["report_sim"]))
    df_th = df_pseudo[df_pseudo["feature_path"].apply(lambda p: s4_map.get(p, 0) > th)]
    print(f"  S4 th={th}: {len(df_th)}")

    # Build D_total
    if cfg.strategy == "Baseline":
        df_dt = df_train[["feature_path", "label"]]
    elif cfg.strategy == "Image":
        df_dt = build_dtotal_raw(df_train, df_pseudo)
    elif cfg.strategy == "Image+FP":
        df_dt = build_dtotal_calib(df_train, df_pseudo, proto_0, proto_1,
                                   sp30_0, sp70_0, sp30_1, sp70_1,
                                   os.path.join(out_dir, "calib"))
    elif cfg.strategy == "Image+Text":
        df_dt = build_dtotal_raw(df_train, df_th)
    elif cfg.strategy == "Image+FP+Text":
        calib_all = os.path.join(out_dir, "calib_all"); os.makedirs(calib_all, exist_ok=True)
        fp_rows = []
        for _, r in tqdm(df_pseudo.iterrows(), desc="  FP-all", total=len(df_pseudo)):
            fp = r["feature_path"]
            if not os.path.exists(fp): continue
            feat = torch.load(fp, map_location="cpu", weights_only=False).float()
            lbl = int(r["label"]); proto = proto_0 if lbl == 0 else proto_1
            s = F.cosine_similarity(feat.unsqueeze(0), proto.unsqueeze(0)).item()
            th_ = sp70_0 if lbl == 0 else sp70_1; tl_ = sp30_0 if lbl == 0 else sp30_1
            alpha = max(0.0, min(0.5, 0.5 * (th_ - s) / (th_ - tl_ + 1e-8)))
            if alpha > 0: feat = (1 - alpha) * feat + alpha * proto
            sp = os.path.join(calib_all, os.path.basename(fp)); torch.save(feat, sp)
            fp_rows.append({"orig_path": fp, "calib_path": sp, "label": lbl})
        df_fp = pd.DataFrame(fp_rows)
        s4_set = set(df_th["feature_path"])
        df_fp_filt = df_fp[df_fp["orig_path"].isin(s4_set)]
        dt_rows = [{"feature_path": r["feature_path"], "label": r["label"]}
                   for _, r in df_train.iterrows()]
        for _, r in df_fp_filt.iterrows():
            dt_rows.append({"feature_path": r["calib_path"], "label": int(r["label"])})
        df_dt = pd.DataFrame(dt_rows)
    elif cfg.strategy == "Image+Text+FP":
        df_dt = build_dtotal_calib(df_train, df_th, proto_0, proto_1,
                                   sp30_0, sp70_0, sp30_1, sp70_1,
                                   os.path.join(out_dir, "calib"))

    print(f"  D_total={len(df_dt)}")
    auc = train_eval(df_dt, test_csv, tag, out_dir)
    print(f"Done: AUC={auc:.4f}")
