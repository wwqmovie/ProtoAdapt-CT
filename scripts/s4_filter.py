"""
S4: Cross-lingual report consistency filtering.

Usage:
    python scripts/s4_filter.py \
        --pseudo_csv data/pseudo_labels.csv \
        --ct_mapping data/ct_report_mapping.csv \
        --lamed_path /path/to/lammed \
        --output data/pseudo_labels_filtered.csv \
        --threshold 0.60
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from protoadapt.text.report_similarity import (
    load_lammed, generate_lammed_report, compute_report_similarity, load_sbert,
)
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="S4 report consistency filter")
    parser.add_argument("--pseudo_csv", required=True, help="Input pseudo-labels CSV")
    parser.add_argument("--ct_mapping", required=True,
                        help="CSV with columns: local_feature_path, ct_dir, report_txt")
    parser.add_argument("--lamed_path", required=True,
                        help="Path to LaMed model checkpoint directory")
    parser.add_argument("--sbert_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="SBERT model name or path")
    parser.add_argument("--output", required=True, help="Output CSV with report_sim column")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Minimum report_sim to keep (default 0.60)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    df_pseudo = pd.read_csv(args.pseudo_csv)
    mapping = pd.read_csv(args.ct_mapping)
    ct_map = dict(zip(mapping["local_feature_path"], mapping["ct_dir"]))
    rpt_map = dict(zip(mapping["ct_dir"], mapping["report_txt"]))

    # Find pseudo-labeled samples that have CT→report pairs
    pseudo_paths = set(df_pseudo["feature_path"])
    scored = 0
    results = []

    lamed_m, lamed_tok = load_lammed(args.lamed_path, device=args.device)
    sbert_m = load_sbert(args.sbert_model)

    for _, row in tqdm(df_pseudo.iterrows(), total=len(df_pseudo), desc="S4"):
        fp = row["feature_path"]
        if fp not in ct_map:
            continue
        ct_dir = ct_map[fp]
        if ct_dir not in rpt_map:
            continue
        rpt_path = rpt_map[ct_dir]
        if not os.path.exists(rpt_path):
            continue

        with open(rpt_path, encoding="utf-8") as f:
            original = f.read().strip()

        try:
            generated = generate_lammed_report(ct_dir, lamed_m, lamed_tok)
        except Exception:
            continue

        sim = compute_report_similarity(generated, original, sbert_m)
        results.append({**row.to_dict(), "generated_report": generated,
                        "gt_report": original, "report_sim": sim})
        scored += 1

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output, index=False)

    n_pass = (df_out["report_sim"] > args.threshold).sum()
    print(f"S4 done: {n_pass}/{len(df_out)} passed (threshold={args.threshold}, scored={scored})")


if __name__ == "__main__":
    main()
