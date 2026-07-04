"""
ProtoAdapt-CT: Full pipeline (Image + Text + FP).

End-to-end training with prototype-guided pseudo-labeling,
cross-lingual report consistency filtering, and feature calibration.

Usage:
    python main.py \
        --train_csv data/labeled_train.csv \
        --test_csv data/labeled_test.csv \
        --pretrain_dir data/features/ \
        --ct_mapping data/ct_report_mapping.csv \
        --lamed_path /path/to/lammed \
        --output_dir output/
"""

import argparse, os, sys, random, hashlib
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protoadapt.models import ResNet1D
from protoadapt.data_loader import prepare_dataset
from protoadapt.engine import train_one_epoch, evaluate
from protoadapt.optim import create_optimizer
from protoadapt.utils import NativeScalerWithGradNormCount, cosine_scheduler


def build_prototypes(train_csv):
    df = pd.read_csv(train_csv)
    feats_0, feats_1 = [], []
    for _, r in df.iterrows():
        f = torch.load(r["feature_path"], map_location="cpu", weights_only=False).float()
        (feats_0 if int(r["label"]) == 0 else feats_1).append(f)
    proto_0 = torch.stack(feats_0).mean(0)
    proto_1 = torch.stack(feats_1).mean(0)
    sims_0 = np.array([F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item() for f in feats_0])
    sims_1 = np.array([F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item() for f in feats_1])
    return proto_0, proto_1, sims_0, sims_1


def assign_pseudo(pretrain_dir, proto_0, proto_1, n_sample=100000, n_tau=20000):
    all_pt = sorted([f for f in os.listdir(pretrain_dir) if f.endswith(".pt")])
    spl = random.sample(all_pt, min(n_tau, len(all_pt)))
    s0s, s1s = [], []
    for fn in spl:
        f = torch.load(os.path.join(pretrain_dir, fn), map_location="cpu", weights_only=False).float()
        s0s.append(F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item())
        s1s.append(F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item())
    s0s, s1s = np.array(s0s), np.array(s1s)
    tau = np.percentile(np.maximum(s0s, s1s), 80)
    use = random.sample(all_pt, min(n_sample, len(all_pt)))
    rows = []
    for fn in tqdm(use, desc="Pseudo-labeling"):
        fp = os.path.join(pretrain_dir, fn)
        f = torch.load(fp, map_location="cpu", weights_only=False).float()
        s0 = F.cosine_similarity(f.unsqueeze(0), proto_0.unsqueeze(0)).item()
        s1 = F.cosine_similarity(f.unsqueeze(0), proto_1.unsqueeze(0)).item()
        if max(s0, s1) > tau:
            rows.append({"feature_path": fp, "label": 0 if s0 > s1 else 1})
    return pd.DataFrame(rows), s0s, s1s


def s4_filter(df_pseudo, ct_mapping_csv, lamed_path, threshold=0.60):
    from protoadapt.text.report_similarity import (
        load_lammed, generate_lammed_report, compute_report_similarity, load_sbert,
    )
    mapping = pd.read_csv(ct_mapping_csv)
    ct_map = dict(zip(mapping["local_feature_path"], mapping["ct_dir"]))
    rpt_map = dict(zip(mapping["ct_dir"], mapping["report_txt"]))
    lamed_m, lamed_tok = load_lammed(lamed_path)
    sbert_m = load_sbert()
    results = []
    for _, row in tqdm(df_pseudo.iterrows(), total=len(df_pseudo), desc="S4 filter"):
        fp = row["feature_path"]
        if fp not in ct_map: continue
        ct_dir = ct_map[fp]
        if ct_dir not in rpt_map: continue
        rpt_path = rpt_map[ct_dir]
        if not os.path.exists(rpt_path): continue
        with open(rpt_path, encoding="utf-8") as fh:
            original = fh.read().strip()
        try:
            generated = generate_lammed_report(ct_dir, lamed_m, lamed_tok)
        except Exception:
            continue
        sim = compute_report_similarity(generated, original, sbert_m)
        if sim > threshold:
            results.append({**row.to_dict(), "report_sim": sim})
    return pd.DataFrame(results)


def calibrate_and_build_dtotal(df_train, df_pseudo, proto_0, proto_1, s0s, s1s, calib_dir):
    os.makedirs(calib_dir, exist_ok=True)
    s0p30, s0p70 = float(np.percentile(s0s, 30)), float(np.percentile(s0s, 70))
    s1p30, s1p70 = float(np.percentile(s1s, 30)), float(np.percentile(s1s, 70))
    rows = [{"feature_path": r["feature_path"], "label": r["label"]}
            for _, r in df_train.iterrows()]
    for _, r in df_pseudo.iterrows():
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


def train(df_dt, test_csv, output_dir, epochs=200, lr=4e-3, batch_size=256, device="cuda"):
    tmp_tr = os.path.join(output_dir, "_train.csv")
    tmp_te = os.path.join(output_dir, "_test.csv")
    df_dt.to_csv(tmp_tr, index=False); pd.read_csv(test_csv).to_csv(tmp_te, index=False)

    class A: pass
    args = A()
    args.train_path=tmp_tr; args.val_path=tmp_te
    args.column_names="feature_path"; args.labels_name="label"
    args.batch_size=batch_size; args.num_workers=0; args.pin_mem=False; args.seed=42
    args.epochs=epochs; args.lr=lr; args.min_lr=1e-8
    args.warmup_epochs=20; args.warmup_steps=-1
    args.weight_decay=0.05; args.weight_decay_end=None
    args.update_freq=1; args.clip_grad=None; args.use_amp=False
    args.distributed=False; args.save_ckpt=False; args.log_dir=None
    args.start_epoch=0; args.device=device
    args.opt="adamw"; args.opt_eps=1e-8; args.opt_betas=None
    args.momentum=0.9; args.layer_decay=1.0; args.output_dir=output_dir
    args.pre_path=""; args.pull_strength=0.4

    dl_tr, dl_val, n_tr, _ = prepare_dataset(args, num_tasks=1, global_rank=0)
    model = ResNet1D(input_dim=400, num_classes=2).to(device)
    opt = create_optimizer(args, model)
    crit = torch.nn.CrossEntropyLoss()
    scaler = NativeScalerWithGradNormCount()
    spe = max(1, n_tr // batch_size)
    warmup = min(epochs - 1, 20)
    lr_s = cosine_scheduler(lr, 1e-8, epochs, spe, warmup_epochs=warmup)
    wd_s = cosine_scheduler(0.05, 0.05, epochs, spe, warmup_epochs=warmup)

    best_auc, best_state = 0.0, None
    for ep in range(epochs):
        train_one_epoch(model, crit, dl_tr, opt, device, ep, scaler, None, None, None,
                        start_steps=ep*spe, lr_schedule_values=lr_s, wd_schedule_values=wd_s,
                        num_training_steps_per_epoch=spe, update_freq=1, use_amp=False)
        ts = evaluate(dl_val, model, device)
        if ts["auc"] > best_auc:
            best_auc = ts["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "auc": best_auc},
                   os.path.join(output_dir, "best_model.pth"))
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for b in dl_val:
            o = model(b["features"].to(device))
            ap.extend(F.softmax(o, dim=1)[:, 1].cpu().numpy())
            al.extend(b["labels"].numpy())
    ap, al = np.array(ap), np.array(al)
    df_te = pd.read_csv(test_csv)
    n = min(len(df_te), len(al))
    df_pred = pd.DataFrame({
        "feature_path": df_te["feature_path"].values[:n], "label": al[:n].astype(int),
        "prob_class0": 1-ap[:n], "prob_class1": ap[:n], "pred": (ap[:n]>=0.5).astype(int),
    })
    df_pred.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    print(f"Best AUC: {best_auc:.4f}")
    return best_auc


def main():
    parser = argparse.ArgumentParser(description="ProtoAdapt-CT full pipeline")
    parser.add_argument("--task", default=None,
                        help="Task name (auto-finds data/downstream/{task}/train.csv)")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--pretrain_dir", required=True)
    parser.add_argument("--ct_mapping", default=None, help="CSV for S4 CT→report mapping")
    parser.add_argument("--lamed_path", default=None, help="LaMed checkpoint path")
    parser.add_argument("--s4_threshold", type=float, default=0.60)
    parser.add_argument("--output_dir", default="output/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve train/test paths from --task or --train_csv/--test_csv
    if args.task:
        train_csv = f"data/downstream/{args.task}/train.csv"
        test_csv = f"data/downstream/{args.task}/test.csv"
    else:
        train_csv, test_csv = args.train_csv, args.test_csv
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    # Auto-map CT paths to feature paths if needed
    ct_col = None
    for c in ["img_path", "ct_path"]:
        if c in df_train.columns:
            ct_col = c; break
    if ct_col:
        def _feat(p):
            return f"data/downstream/features/{hashlib.md5(str(p).encode()).hexdigest()}.pt"
        df_train["feature_path"] = df_train[ct_col].apply(_feat)
        df_test["feature_path"] = df_test[ct_col].apply(_feat)
        df_train = df_train.drop(columns=[ct_col])
        df_test = df_test.drop(columns=[ct_col])
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # S1-S3: Prototypes + pseudo-labels
    proto_0, proto_1, s0s, s1s = build_prototypes(train_csv)
    pcos = F.cosine_similarity(proto_0.unsqueeze(0), proto_1.unsqueeze(0)).item()
    print(f"Prototypes: cos_sim={pcos:.4f}")

    df_pseudo, pt_s0s, pt_s1s = assign_pseudo(args.pretrain_dir, proto_0, proto_1)
    print(f"Pseudo-labels: {len(df_pseudo)}")

    # S4: Report filtering (if mapping + LaMed provided)
    if args.ct_mapping and args.lamed_path:
        df_pseudo = s4_filter(df_pseudo, args.ct_mapping, args.lamed_path, args.s4_threshold)
        print(f"After S4: {len(df_pseudo)}")

    # FP: Feature calibration + D_total
    calib_dir = os.path.join(args.output_dir, "calibrated")
    df_dtotal = calibrate_and_build_dtotal(df_train, df_pseudo, proto_0, proto_1,
                                           pt_s0s, pt_s1s, calib_dir)
    print(f"D_total: {len(df_dtotal)} (train={len(df_train)}+pseudo={len(df_pseudo)})")

    # Train
    auc = train(df_dtotal, test_csv, args.output_dir,
                epochs=args.epochs, lr=args.lr, device=args.device)
    print(f"Done: AUC={auc:.4f}")


if __name__ == "__main__":
    main()
