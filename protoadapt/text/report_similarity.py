"""
S4: Cross-lingual report consistency filtering.

Pipeline:
    1. LaMed generates English radiology report from CT image.
    2. SBERT computes cosine similarity between generated (EN) and original (ZH) reports.
    3. Pseudo-labels with report_sim < threshold are filtered out.

Requires:
    - LaMed checkpoint (M3D-LaMed)
    - SBERT model (paraphrase-multilingual-MiniLM-L12-v2)
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


def load_ct_for_lammed(ct_dir):
    """Load CT from directory and preprocess for LaMed input.

    LaMed expects (1, 3, 48, 256, 256) with 192 image patches.
    Same preprocessing as Swin3D feature extraction.

    Args:
        ct_dir: directory containing CT slices (.nii.gz or DICOM)

    Returns:
        torch.Tensor of shape (1, 3, 48, 256, 256) in bfloat16
    """
    import SimpleITK as sitk
    import cv2

    # Find image file
    img_path = None
    for fname in os.listdir(ct_dir):
        if fname.endswith((".nii.gz", ".nii")):
            img_path = os.path.join(ct_dir, fname)
            break
    if img_path is None:
        # Try DICOM
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
        if dicom_names:
            reader.SetFileNames(dicom_names)
            img = reader.Execute()
        else:
            raise FileNotFoundError(f"No readable image found in {ct_dir}")
    else:
        img = sitk.ReadImage(img_path)

    data = sitk.GetArrayFromImage(img).astype(np.float32)
    data = np.clip(data, -1500, 500) + 1500
    data = data / (data.max() + 1e-8)

    target = (48, 256, 256)
    resized = np.zeros(target, dtype=np.float32)
    for i in range(min(target[0], data.shape[0])):
        resized[i] = cv2.resize(data[i], (target[2], target[1]))

    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    tensor = tensor.repeat(1, 3, 1, 1, 1)
    return tensor


def load_lammed(model_path, device="cuda"):
    """Load LaMed model and tokenizer.

    Args:
        model_path: path to LaMed checkpoint directory
        device: torch device (default "cuda")

    Returns:
        model, tokenizer
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=512, padding_side="right",
        trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    return model, tokenizer


def generate_lammed_report(ct_dir, model, tokenizer):
    """Generate English radiology report from CT using LaMed.

    Args:
        ct_dir: directory containing CT slices
        model: LaMed model
        tokenizer: LaMed tokenizer

    Returns:
        str: generated English report text, cleaned
    """
    image = load_ct_for_lammed(ct_dir).cuda()

    question = "What details are displayed in this CT scan image?"
    image_tokens = "<im_patch>" * 192
    input_txt = image_tokens + question
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].cuda()
    attn_mask = tokenizer(input_txt, return_tensors="pt")["attention_mask"].cuda()

    with torch.no_grad():
        generation = model.generate(
            image, input_ids, attention_mask=attn_mask,
            max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    text = text.replace("\n", "").strip()

    # Clean special tokens
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_sbert(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Load multilingual SBERT model for cross-lingual similarity.

    Args:
        model_name: SBERT model name on HuggingFace or local path

    Returns:
        SentenceTransformer model (on CUDA if available)
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def compute_report_similarity(generated_en, original_zh, sbert_model):
    """Compute cosine similarity between generated English and original Chinese reports.

    Args:
        generated_en: English report text from LaMed
        original_zh: original Chinese report text
        sbert_model: SentenceTransformer model

    Returns:
        float: cosine similarity (0 to 1)
    """
    embeddings = sbert_model.encode(
        [generated_en, original_zh], convert_to_tensor=True
    )
    sim = F.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    )
    return max(0.0, float(sim.item()))


def filter_pseudo_labels(pseudo_csv, ct_dir_mapping, report_mapping,
                         lamed_model_path, sbert_model_name,
                         output_csv, threshold=0.60):
    """Run full S4 report consistency filtering pipeline.

    Args:
        pseudo_csv: input CSV with pseudo-labels (must have feature_path column)
        ct_dir_mapping: dict mapping MD5 (from feature_path) -> CT directory path
        report_mapping: dict mapping CT directory -> original report .txt path
        lamed_model_path: path to LaMed checkpoint directory
        sbert_model_name: SBERT model name or local path
        output_csv: path to save filtered results
        threshold: minimum report_sim to keep (default 0.60)

    Returns:
        pd.DataFrame: pseudo-labels with added report_sim column
    """
    df = pd.read_csv(pseudo_csv)
    model, tokenizer = load_lammed(lamed_model_path)
    sbert = load_sbert(sbert_model_name)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="S4 filtering"):
        fp = row["feature_path"]
        md5 = os.path.splitext(os.path.basename(fp))[0]

        ct_dir = ct_dir_mapping.get(md5)
        if ct_dir is None or not os.path.isdir(ct_dir):
            continue

        report_txt = report_mapping.get(ct_dir)
        if report_txt is None or not os.path.exists(report_txt):
            continue

        with open(report_txt, "r", encoding="utf-8") as f:
            gt_report = f.read().strip()
        if not gt_report:
            continue

        try:
            generated = generate_lammed_report(ct_dir, model, tokenizer)
        except Exception:
            continue

        sim = compute_report_similarity(generated, gt_report, sbert)

        results.append({
            **{k: v for k, v in row.items()},
            "generated_report": generated,
            "gt_report": gt_report,
            "report_sim": sim,
        })

    df_out = pd.DataFrame(results)
    df_filtered = df_out[df_out["report_sim"] > threshold]

    df_out.to_csv(output_csv, index=False)
    print(f"S4 done: {len(df_filtered)}/{len(df_out)} passed "
          f"(threshold={threshold}, {len(results)} scored)")
    return df_filtered
