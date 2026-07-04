"""
Build prototypes and assign pseudo-labels from pretrain features.

Usage:
    python scripts/assign_pseudo_labels.py \
        --train_csv data/labeled_train.csv \
        --pretrain_dir data/pretrain_features/ \
        --output data/pseudo_labels.csv \
        --method top20
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protoadapt.pseudo_label import build_prototypes, top20_pseudo


def main():
    parser = argparse.ArgumentParser(description="Assign pseudo-labels to pretrain features")
    parser.add_argument("--train_csv", required=True, help="Labeled training CSV (feature_path, label)")
    parser.add_argument("--pretrain_dir", required=True, help="Directory of pretrain .pt features")
    parser.add_argument("--output", required=True, help="Output CSV for pseudo-labels")
    parser.add_argument("--method", default="top20", choices=["top20"],
                        help="Pseudo-labeling method")
    parser.add_argument("--n_sample", type=int, default=100000,
                        help="Number of pretrain samples to process")
    parser.add_argument("--n_tau", type=int, default=20000,
                        help="Number of samples for tau estimation")
    parser.add_argument("--proto_out", default=None,
                        help="Directory to save prototypes (class_0_average.pt, class_1_average.pt)")
    args = parser.parse_args()

    proto_0, proto_1, stats = build_prototypes(args.train_csv)
    print(f"Prototypes: cos_sim={stats['cos_sim']:.4f}")

    if args.proto_out:
        import torch
        os.makedirs(args.proto_out, exist_ok=True)
        torch.save(proto_0, os.path.join(args.proto_out, "class_0_average.pt"))
        torch.save(proto_1, os.path.join(args.proto_out, "class_1_average.pt"))
        print(f"Saved prototypes to {args.proto_out}")

    df_pseudo, tau_stats = top20_pseudo(
        args.pretrain_dir, proto_0, proto_1,
        n_sample=args.n_sample, n_tau=args.n_tau,
    )
    df_pseudo.to_csv(args.output, index=False)
    print(f"Pseudo-labels: {len(df_pseudo)} saved to {args.output}")
    print(f"Tau: {tau_stats['tau']:.4f}")


if __name__ == "__main__":
    main()
