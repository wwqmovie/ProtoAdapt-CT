"""
Extract 400-dim features from 3D CT volumes using frozen Swin3D backbone.

Input: CT volumes (.nii.gz or DICOM dir) + CSV with img_path, label (label optional)
Output: .pt feature files (MD5-hash named) in output_dir

Usage:
    python scripts/extract_features.py --input_csv ct_list.csv --output_dir features/
    python scripts/extract_features.py --input_csv ct_list.csv --output_dir features/ --pretrained /path/to/checkpoint.safetensors
"""

import argparse
import hashlib
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_and_preprocess_ct(img_path):
    """Load a 3D CT volume and preprocess for Swin3D input.

    Supports .nii.gz files and DICOM directories.
    Returns tensor of shape (1, 3, 48, 256, 256).

    Args:
        img_path: path to .nii.gz file or DICOM directory

    Returns:
        torch.Tensor of shape (1, 3, 48, 256, 256), normalized
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("SimpleITK required. pip install SimpleITK")

    if os.path.isdir(img_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(img_path)
        if not dicom_names:
            raise FileNotFoundError(f"No DICOM files found in {img_path}")
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(img_path)

    data = sitk.GetArrayFromImage(img).astype(np.float32)

    # Clip and normalize
    data = np.clip(data, -1500, 500) + 1500
    data = data / (data.max() + 1e-8)

    # Resize to (48, 256, 256)
    import cv2
    target = (48, 256, 256)
    resized = np.zeros(target, dtype=np.float32)
    for i in range(min(target[0], data.shape[0])):
        resized[i] = cv2.resize(data[i], (target[2], target[1]))

    tensor = torch.from_numpy(resized).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    return tensor


def build_model(pretrained_path, device="cuda"):
    """Build Swin3D feature extractor.

    Loads torchvision swin3d_b with Kinetics-400 head (400-dim output),
    optionally with CLIP-pretrained vision encoder weights.

    Args:
        pretrained_path: path to CLIP pretrained .safetensors (optional, set None to skip)
        device: torch device

    Returns:
        Swin3D model in eval mode
    """
    from protoadapt.models.swin3d import Swin3D

    class Args:
        in_channels = 3
        num_classes = 2
        freeze = "all+feature"
        head = "default"
        pretrained = pretrained_path
        patch_size = [2, 4, 4]
        window_size = [4, 7, 7]
        depths = [2, 2, 2]
        embed_dim = 128
        drop_path_rate = 0.0
        img_size = 256

    args = Args()
    model = Swin3D(args=args).to(device)
    model.eval()
    return model


def extract_features(input_csv, output_dir, pretrained=None, batch_size=1, device="cuda"):
    """Extract Swin3D features for all CTs listed in input_csv.

    Args:
        input_csv: CSV with columns 'img_path' and optionally 'label'
        output_dir: directory to save .pt feature files (MD5-named)
        pretrained: path to CLIP-pretrained checkpoint (None for Kinetics-400 only)
        batch_size: inference batch size (default 1 for 3D)
        device: torch device
    """
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    model = build_model(pretrained, device)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        img_path = row["img_path"]
        md5 = hashlib.md5(img_path.encode()).hexdigest()
        out_path = os.path.join(output_dir, f"{md5}.pt")

        if os.path.exists(out_path):
            continue

        ct_tensor = load_and_preprocess_ct(img_path).to(device)

        with torch.no_grad():
            feat = model(ct_tensor)  # (1, 400)

        torch.save(feat.squeeze(0).cpu(), out_path)

    print(f"Done: {len(df)} features saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin3D feature extraction")
    parser.add_argument("--input_csv", required=True,
                        help="CSV with img_path column")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save .pt feature files")
    parser.add_argument("--pretrained", default=None,
                        help="Path to CLIP-pretrained checkpoint .safetensors")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_features(args.input_csv, args.output_dir,
                     args.pretrained, args.batch_size, args.device)
