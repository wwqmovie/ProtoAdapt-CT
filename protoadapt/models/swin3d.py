"""
swin3d.py — Swin3D vision encoder for CT feature extraction (reconstructed).

 feature_extract.py ,  3D CT  400-dim 。
 torchvision swin3d_b  backbone,  CLIP  vision encoder ,
 Kinetics-400  (1024→400)  400-dim 。

:
    from model.swin3d import Swin3D, Swin3DforPretrain
    model = Swin3D(args=args)
    features = model(images)  # (B, 400)
"""

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from safetensors.torch import load_file
import os


class Swin3D(nn.Module):
    """ 3D CT  400-dim  Swin3D """

    def __init__(self, args=None):
        super().__init__()
        self.args = args

        # 1.  torchvision swin3d_b ( Kinetics-400 )
        try:
            #  Kinetics-400  ( head)
            weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
            self.model = swin3d_b(weights=weights)
        except Exception:
            # , 
            self.model = swin3d_b(weights=None)

        # 2.  CLIP pretrained ,  vision encoder 
        if args is not None and hasattr(args, 'pretrained') and args.pretrained:
            pretrained_path = args.pretrained
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                self._load_pretrained(pretrained_path)

        self.hidden_size = 400  # Kinetics-400 

    def _load_pretrained(self, path):
        """ CLIP safetensors  .bin checkpoint  vision_encoder """
        if path.endswith('.safetensors'):
            state_dict = load_file(path, device='cpu')
        else:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

        #  vision_encoder.model.* 
        new_sd = {}
        for k, v in state_dict.items():
            # safetensors : vision_encoder.model.*
            if k.startswith('vision_encoder.model.'):
                new_sd[k.replace('vision_encoder.model.', '')] = v
            # .bin : model.* ( vision_encoder )
            elif k.startswith('model.'):
                new_sd[k] = v
            #  ( checkpoint)
            elif k.startswith('features.') or k.startswith('patch_embed.') or k.startswith('norm.'):
                new_sd[k] = v

        if new_sd:
            #  self.model (torchvision SwinTransformer3d)
            #  backbone ,  Kinetics head
            model_sd = self.model.state_dict()
            matched = {k: v for k, v in new_sd.items()
                      if k in model_sd and model_sd[k].shape == v.shape}
            model_sd.update(matched)
            self.model.load_state_dict(model_sd, strict=True)
            print(f"  Loaded {len(matched)}/{len(new_sd)} backbone keys")
        else:
            print(f"  Warning: No matching keys found in {path}")

    def forward(self, x):
        """x: (B, 3, D, H, W) → (B, 400)"""
        return self.model(x)


class Swin3DforPretrain(nn.Module):
    """ Swin3D,  (B, N, 1024)  ()"""

    def __init__(self, args=None):
        super().__init__()
        self.args = args

        self.model = swin3d_b(weights=None)
        self.model.head = nn.Identity()
        self.model.avgpool = nn.Identity()

        if args is not None and hasattr(args, 'pretrained') and args.pretrained:
            pretrained_path = args.pretrained
            if os.path.exists(pretrained_path):
                self._load_pretrained(pretrained_path)

        self._hidden_size = self.model.num_features  # 1024

    def _load_pretrained(self, path):
        """ Swin3D._load_pretrained"""
        if path.endswith('.safetensors'):
            state_dict = load_file(path, device='cpu')
        else:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('vision_encoder.model.'):
                new_sd[k.replace('vision_encoder.model.', '')] = v
            elif k.startswith('model.'):
                new_sd[k] = v
            elif k.startswith('features.') or k.startswith('patch_embed.') or k.startswith('norm.'):
                new_sd[k] = v

        if new_sd:
            model_sd = self.model.state_dict()
            matched = {k: v for k, v in new_sd.items()
                      if k in model_sd and model_sd[k].shape == v.shape}
            model_sd.update(matched)
            self.model.load_state_dict(model_sd, strict=False)
            print(f"  Loaded {len(matched)}/{len(new_sd)} backbone keys")

    @property
    def hidden_size(self):
        return self.model.num_features

    def forward(self, x):
        x = self.model(x)  # (B, 1024, D', H', W')
        x = torch.reshape(x, (x.shape[0], self.hidden_size, -1))
        return x.permute(0, 2, 1)  # (B, N, 1024)


if __name__ == '__main__':
    # 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str,
                        default="")
    args = parser.parse_args([])  # empty args for testing

    model = Swin3D(args=args)
    x = torch.randn(1, 3, 48, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Feature dim: {out.shape[-1]}")
