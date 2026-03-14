"""export_clean_components.py -- Export clean backbone and MADDN weights after LoRA merging.

Usage (PowerShell):
    # Recommended: export from already-merged model
    python export_clean_components.py --merged ./checkpoints/model_merged_clone.pth

    # Fallback: export from best checkpoint (auto-merge LoRA temporarily)
    python export_clean_components.py --best ./checkpoints/model_best.pth.tar

    # Custom output directory
    python export_clean_components.py --merged ./checkpoints/model_merged_clone.pth --outdir ./checkpoints/clean_components

Output files:
    - backbone_finetuned_clean.pth      (3D ResNet weights only)
    - maddn_finetuned_clean.pth         (MADDN fusion_network weights only)
    - classifier_finetuned_clean.pth    (classifier head weights only)
"""

import os
import argparse
import torch

from config import Config
from model_builder import build_maddn_net


def load_state_dict_loose(model, state):
    """Load state_dict loosely, ignoring shape mismatches."""
    model_dict = model.state_dict()
    loaded, skipped = [], []
    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k].copy_(v)
            loaded.append(k)
        else:
            skipped.append(k)
    return loaded, skipped


def try_merge_lora_inplace(model):
    """Merge LoRA weights into base weights in-place.

    Assumes LoRALinear / LoRAConv3d have attributes: base, lora_A, lora_B, scaling.
    """
    replaced = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in ("LoRALinear", "LoRAConv3d"):
            if not (hasattr(module, 'base') and hasattr(module, 'lora_A')
                    and hasattr(module, 'lora_B')):
                continue
            with torch.no_grad():
                if cls_name == "LoRALinear":
                    delta = module.lora_B.weight @ module.lora_A.weight
                    module.base.weight += delta * module.scaling
                else:
                    B = module.lora_B.weight.view(
                        module.lora_B.weight.shape[0],
                        module.lora_B.weight.shape[1])
                    A = module.lora_A.weight.view(
                        module.lora_A.weight.shape[0], -1)
                    delta = (B @ A).view_as(module.base.weight)
                    module.base.weight += delta * module.scaling
            replaced += 1
    return replaced


def extract_and_save_components(state_dict, outdir):
    """Split state_dict into backbone / maddn-fusion / classifier and save."""
    os.makedirs(outdir, exist_ok=True)
    backbone_state = {}
    maddn_state = {}
    classifier_state = {}

    for k, v in state_dict.items():
        # Skip LoRA residual keys (should not exist in merged model)
        if 'lora_' in k or '.base.' in k:
            continue
        if k.startswith('backbone.'):
            backbone_state[k.replace('backbone.', '')] = v
        elif (k.startswith('fusion_network.')
              or k.startswith('shared_transformer.')):
            maddn_state[k] = v
        elif 'classifier.' in k or k.startswith('fc.'):
            classifier_state[k] = v

    bb_path = os.path.join(outdir, 'backbone_finetuned_clean.pth')
    maddn_path = os.path.join(outdir, 'maddn_finetuned_clean.pth')
    cls_path = os.path.join(outdir, 'classifier_finetuned_clean.pth')

    if backbone_state:
        torch.save({'state_dict': backbone_state}, bb_path)
        print(f"[Export] backbone: {bb_path}  (layers={len(backbone_state)})")
    else:
        print("[Export][Warn] No backbone.* keys found.")

    if maddn_state:
        torch.save({'state_dict': maddn_state}, maddn_path)
        print(f"[Export] MADDN fusion: {maddn_path}  (layers={len(maddn_state)})")
    else:
        print("[Export][Warn] No MADDN fusion keys found.")

    if classifier_state:
        torch.save({'state_dict': classifier_state}, cls_path)
        print(f"[Export] classifier: {cls_path}  (layers={len(classifier_state)})")
    else:
        print("[Export][Warn] No classifier keys found.")


def main():
    parser = argparse.ArgumentParser(
        description='Export clean MADDN component weights after LoRA merging.')
    parser.add_argument(
        '--merged',
        default='./checkpoints/model_merged_clone.pth',
        help='Path to already-merged model (preferred)')
    parser.add_argument(
        '--best',
        default='./checkpoints/model_best.pth.tar',
        help='Path to best checkpoint with LoRA (fallback)')
    parser.add_argument(
        '--outdir',
        default='./checkpoints/clean_components',
        help='Output directory for clean component weights')
    args = parser.parse_args()

    config = Config()

    # Prefer the already-merged model
    if os.path.isfile(args.merged):
        print(f"[Info] Loading merged model: {args.merged}")
        merged_ckpt = torch.load(args.merged, map_location='cpu')
        state = (merged_ckpt['state_dict']
                 if 'state_dict' in merged_ckpt else merged_ckpt)
        extract_and_save_components(state, args.outdir)
        return

    # Fallback: load best checkpoint and merge LoRA temporarily
    if not os.path.isfile(args.best):
        print(f"[Error] Neither {args.merged} nor {args.best} found.")
        return

    print(f"[Fallback] Merging LoRA from best checkpoint: {args.best}")
    best_ckpt = torch.load(args.best, map_location='cpu')
    best_state = (best_ckpt['state_dict']
                  if 'state_dict' in best_ckpt else best_ckpt)

    config.lora.enable_lora = True
    model = build_maddn_net(config)
    model.load_state_dict(best_state, strict=False)
    replaced = try_merge_lora_inplace(model)
    print(f"[Merge] Merged {replaced} LoRA modules")
    extract_and_save_components(model.state_dict(), args.outdir)


if __name__ == '__main__':
    main()
