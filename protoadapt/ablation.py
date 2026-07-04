"""
Component ablation runner for Lung1/HUAXI (paper 3-tier S3 pseudo-labels).

Usage:
    python -m protoadapt.ablation --task LUNG1 --strategy Image+Text --test 63 --th 0.55 --gpu 0

Strategies: Baseline, Image, Image+FP, Image+Text, Image+FP+Text, Image+Text+FP
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

from protoadapt.models import ResNet1D, ResNet1DEasy
from protoadapt.data_loader import prepare_dataset
from protoadapt.engine import train_one_epoch, evaluate
from protoadapt.optim import create_optimizer
from protoadapt.utils import NativeScalerWithGradNormCount as NativeScaler, cosine_scheduler
from protoadapt.config import DATA, HP, S4_THRESHOLDS, PRETRAIN_FEAT_DIR, OUTPUT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="ProtoAdapt-CT component ablation")
    p.add_argument("--task", default="LUNG1", choices=["LUNG1", "HUAXI"])
    p.add_argument("--strategy", required=True,
                   choices=["Baseline", "Image", "Image+FP", "Image+Text",
                            "Image+FP+Text", "Image+Text+FP"])
    p.add_argument("--test", type=int, default=63, help="test set size (63 or 84 for LUNG1)")
    p.add_argument("--th", type=float, default=None, help="S4 threshold (default: per-task best)")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def make_args(tr, val):
    """Build Args-like namespace for prepare_dataset."""
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


def build_dtotal_raw(df_train, df_pseudo):
    """D_total without feature pull."""
    rows = [{"feature_path": r["feature_path"], "label": r["label"]}
            for _, r in df_train.iterrows()]
    for _, r in df_pseudo.iterrows():
        if os.path.exists(r["feature_path"]):
            rows.append({"feature_path": r["feature_path"], "label": int(r["label"])})
    return pd.DataFrame(rows)


def build_dtotal_calib(df_train, df_pseudo, proto_0, proto_1,
                       pt_s0_p30, pt_s0_p70, pt_s1_p30, pt_s1_p70, calib_dir):
    """D_total with feature pull calibration: f' = (1-alpha)*f + alpha*P_c."""
    os.makedirs(calib_dir, exist_ok=True)
    rows = [{"feature_path": r["feature_path"], "label": r["label"]}
            for _, r in df_train.iterrows()]
    for _, r in df_pseudo.iterrows():
        fp = r["feature_path"]
        if not os.path.exists(fp):
            continue
        feat = torch.load(fp, map_location="cpu", weights_only=False).float()
        lbl = int(r["label"])
        proto = proto_0 if lbl == 0 else proto_1
        s = F.cosine_similarity(feat.unsqueeze(0), proto.unsqueeze(0)).item()
        th = pt_s0_p70 if lbl == 0 else pt_s1_p70
        tl = pt_s0_p30 if lbl == 0 else pt_s1_p30
        alpha = max(0.0, min(0.5, 0.5 * (th - s) / (th - tl + 1e-8)))
        if alpha > 0:
            feat = (1 - alpha) * feat + alpha * proto
        sp = os.path.join(calib_dir, os.path.basename(fp))
        torch.save(feat, sp)
        rows.append({"feature_path": sp, "label": lbl})
    return pd.DataFrame(rows)


def train_eval(df_dt, test_csv, label_str, out_dir):
    """Train ResNet1D and save best model + predictions."""
    tmp_tr = f"/tmp/{label_str}_tr.csv"; tmp_te = f"/tmp/{label_str}_te.csv"
    df_dt.to_csv(tmp_tr, index=False); pd.read_csv(test_csv).to_csv(tmp_te, index=False)
    args = make_args(tmp_tr, tmp_te)
    dl_tr, dl_val, n_tr, _ = prepare_dataset(args, num_tasks=1, global_rank=0)

    model_cls = ResNet1DEasy if os.environ.get("HUAXI_MODEL") else ResNet1D
    model = model_cls(input_dim=400, num_classes=2).to(DEVICE)
    opt = create_optimizer(args, model)
    crit = torch.nn.CrossEntropyLoss()
    scaler = NativeScaler()
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

    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for b in dl_val:
            o = model(b["features"].to(DEVICE))
            ap.extend(F.softmax(o, dim=1)[:, 1].cpu().numpy())
            al.extend(b["labels"].numpy())
    ap, al = np.array(ap), np.array(al)

    df_te = pd.read_csv(test_csv)
    n = min(len(df_te), len(al))
    df_pred = pd.DataFrame({
        "feature_path": df_te["feature_path"].values[:n],
        "label": al[:n].astype(int),
        "prob_class0": 1 - ap[:n], "prob_class1": ap[:n],
        "pred": (ap[:n] >= 0.5).astype(int),
    })
    df_pred.to_csv(os.path.join(out_dir, f"{label_str}.csv"), index=False)
    print(f"  [{label_str}] AUC={max_auc:.4f}")
    return max_auc


def load_data(task, test_size):
    """Load labeled train/test data."""
    cfg = DATA[task.lower()]
    df_tr = pd.read_csv(cfg["train"])
    if task == "HUAXI":
        df_te = pd.read_csv(cfg["val"])
        if "feature_path" in df_te.columns:
            df_train = df_tr[["feature_path", "label"]]
            df_test = df_te[["feature_path", "label"]]
        else:
            df_train = df_tr[["feature_path", "label"]]
            df_test = df_te  # needs column renaming per data
    else:  # LUNG1
        df_te = pd.read_csv(cfg["test"])
        if test_size == 84:
            df_va = pd.read_csv(cfg["val"])
            df_te = pd.concat([df_te, df_va], ignore_index=True)
        train_rows, test_rows = [], []
        for _, r in df_tr.iterrows():
            md5 = hashlib.md5(r["imgPath"].encode()).hexdigest() + ".pt"
            fp = os.path.join(PRETRAIN_FEAT_DIR, "train", md5)
            if os.path.exists(fp):
                train_rows.append({"feature_path": fp, "label": int(r["label"])})
        feat_subdirs = ["test"] if test_size == 63 else ["test", "val"]
        for _, r in df_te.iterrows():
            fp = None
            for sub in feat_subdirs:
                md5 = hashlib.md5(r["imgPath"].encode()).hexdigest() + ".pt"
                cand = os.path.join(PRETRAIN_FEAT_DIR, sub, md5)
                if os.path.exists(cand):
                    fp = cand; break
            if fp:
                test_rows.append({"feature_path": fp, "label": int(r["label"])})
        df_train = pd.DataFrame(train_rows)
        df_test = pd.DataFrame(test_rows)
    return df_train, df_test


def compute_prototypes_and_tau(df_train):
    """Compute class prototypes and Q1/Q3 from labeled training data."""
    proto_0 = torch.stack([torch.load(r["feature_path"], map_location="cpu",
                          weights_only=False).float()
                          for _, r in df_train.iterrows() if r["label"] == 0]).mean(0)
    proto_1 = torch.stack([torch.load(r["feature_path"], map_location="cpu",
                          weights_only=False).float()
                          for _, r in df_train.iterrows() if r["label"] == 1]).mean(0)
    sims_0 = []
    sims_1 = []
    for _, r in df_train.iterrows():
        f = torch.load(r["feature_path"], map_location="cpu", weights_only=False).float()
        sims_0.append(F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item())
        sims_1.append(F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item())
    sims_0, sims_1 = np.array(sims_0), np.array(sims_1)
    pcos = F.cosine_similarity(proto_0.unsqueeze(0), proto_1.unsqueeze(0)).item()
    print(f"  Proto cos_sim: {pcos:.4f}")
    return (proto_0, proto_1,
            float(np.percentile(sims_0, 30)), float(np.percentile(sims_0, 70)),
            float(np.percentile(sims_1, 30)), float(np.percentile(sims_1, 70)))


if __name__ == "__main__":
    cfg = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    if cfg.task == "HUAXI":
        os.environ["HUAXI_MODEL"] = "1"

    th = cfg.th if cfg.th else S4_THRESHOLDS[cfg.task]
    tag = f"{cfg.task}_n{cfg.test}_th{int(th*100):d}_{cfg.strategy.replace('+','_')}"
    out_dir = os.path.join(OUTPUT_DIR, "ablation", tag)
    os.makedirs(out_dir, exist_ok=True)
    random.seed(HP["seed"]); np.random.seed(HP["seed"])

    print(f"Task={cfg.task} Strategy={cfg.strategy} Test={cfg.test} S4_th={th}")

    # Load data
    df_train, df_test = load_data(cfg.task, cfg.test)
    test_csv = f"/tmp/test_{tag}.csv"; df_test.to_csv(test_csv, index=False)
    print(f"  Train={len(df_train)} Test={len(df_test)}")

    # Prototypes and calibration thresholds
    proto_0, proto_1, sp30_0, sp70_0, sp30_1, sp70_1 = compute_prototypes_and_tau(df_train)

    # Load S3 and S4
    dcfg = DATA[cfg.task.lower()]
    df_s3 = pd.read_csv(dcfg["s3_pseudo"])
    s4_map = {}
    if os.path.exists(dcfg.get("s4_full", "")):
        df_s4 = pd.read_csv(dcfg["s4_full"])
        s4_map = dict(zip(df_s4["feature_path"], df_s4["report_sim"]))
    df_th = df_s3[df_s3["feature_path"].apply(lambda p: s4_map.get(p, 0) > th)]
    print(f"  S3={len(df_s3)} S4(th={th})={len(df_th)}")

    # Build D_total
    if cfg.strategy == "Baseline":
        df_dt = df_train[["feature_path", "label"]]
    elif cfg.strategy == "Image":
        df_dt = build_dtotal_raw(df_train, df_s3)
    elif cfg.strategy == "Image+FP":
        df_dt = build_dtotal_calib(df_train, df_s3, proto_0, proto_1,
                                   sp30_0, sp70_0, sp30_1, sp70_1,
                                   os.path.join(out_dir, "calib"))
    elif cfg.strategy == "Image+Text":
        df_dt = build_dtotal_raw(df_train, df_th)
    elif cfg.strategy == "Image+FP+Text":
        # FP -> Text: calibrate ALL S3 features first, then S4 filter
        calib_all = os.path.join(out_dir, "calib_all")
        os.makedirs(calib_all, exist_ok=True)
        fp_rows = []
        for _, r in tqdm(df_s3.iterrows(), desc="  FP-all", total=len(df_s3)):
            fp = r["feature_path"]
            if not os.path.exists(fp): continue
            feat = torch.load(fp, map_location="cpu", weights_only=False).float()
            lbl = int(r["label"]); proto = proto_0 if lbl == 0 else proto_1
            s = F.cosine_similarity(feat.unsqueeze(0), proto.unsqueeze(0)).item()
            th_ = sp70_0 if lbl == 0 else sp70_1
            tl_ = sp30_0 if lbl == 0 else sp30_1
            alpha = max(0.0, min(0.5, 0.5 * (th_ - s) / (th_ - tl_ + 1e-8)))
            if alpha > 0: feat = (1 - alpha) * feat + alpha * proto
            sp = os.path.join(calib_all, os.path.basename(fp))
            torch.save(feat, sp)
            fp_rows.append({"orig_path": fp, "calib_path": sp, "label": lbl})
        df_fp = pd.DataFrame(fp_rows)
        s4_set = set(df_th["feature_path"])
        df_fp_filt = df_fp[df_fp["orig_path"].isin(s4_set)]
        dt_rows = [{"feature_path": r["feature_path"], "label": r["label"]}
                   for _, r in df_train.iterrows()]
        for _, r in df_fp_filt.iterrows():
            dt_rows.append({"feature_path": r["calib_path"], "label": int(r["label"])})
        df_dt = pd.DataFrame(dt_rows)
        print(f"  FP->Text: {len(df_fp_filt)} pseudo (from {len(df_fp)} calibrated)")
    elif cfg.strategy == "Image+Text+FP":
        # Text -> FP: S4 filter first, then calibrate
        df_dt = build_dtotal_calib(df_train, df_th, proto_0, proto_1,
                                   sp30_0, sp70_0, sp30_1, sp70_1,
                                   os.path.join(out_dir, "calib"))

    print(f"  D_total={len(df_dt)} (train={len(df_train)} + pseudo={len(df_dt)-len(df_train)})")
    auc = train_eval(df_dt, test_csv, tag, out_dir)
    print(f"Done: AUC={auc:.4f}")
