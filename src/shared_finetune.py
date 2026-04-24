from .data_utils import *
from .mae_masked import *

import os
import torch
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Assumes these are already defined/importable in your project:
# - MultiOmicDataset, get_dataloader
# - build_pretrain_ae_for_modality, load_modality_with_config, extract_encoder_decoder_from_pretrained
# - MultiModalWithSharedSpace
# - finetune_epoch, eval_finetune_epoch


def run_shared_finetune(
    multi_omic_data: Dict[str, pd.DataFrame],
    common_samples: List[str],
    train_idx: List[int],
    val_idx: List[int],
    test_idx: Optional[List[int]] = None,
    model_paths: Optional[Dict[str, str]] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    shared_dim: int = 128,
    proj_depth: int = 1,
    batch_size: int = 64,
    shuffle_train: bool = True,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    epochs: int = 100,
    lambda_contrast: float = 1.0,
    lambda_impute: float = 1.0,
    lambda_recon: float = 1.0,
    modality_dropout_prob: float = 0.3,
    feature_mask_p_train: float = 0.2,
    feature_mask_p_val: float = 0.2,
    alpha_mask_recon: float = 0.5,          # NEW: weight for masked vs overall recon in Phase 2
    two_path_clean_for_contrast: bool = False,
    freeze_encoders_decoders: bool = False,
    verbose: bool = True,
) -> Tuple[
    "MultiModalWithSharedSpace",
    Dict[str, List[float]],
    Dict[str, List[float]],
    "torch.utils.data.DataLoader",
    "torch.utils.data.DataLoader",
    Optional["torch.utils.data.DataLoader"],
    torch.optim.Optimizer
]:
    """
    Build the shared-space model from pretrained per-modality AEs and finetune it.

    Returns:
        model, train_loss_hist, val_loss_hist, train_loader, val_loader, test_loader, opt
    """

    # 1) Load pretrained modality AEs and extract encoders/decoders + mask_values
    encoders, decoders, hidden_dims = {}, {}, {}
    mask_values: Dict[str, float] = {}

    if model_paths is None:
        model_paths = {m: f"{m}_ae.pt" for m in multi_omic_data.keys()}

    for mod in multi_omic_data.keys():
        path = model_paths[mod]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing pretrained AE for modality '{mod}'. Expected file: {path}")

        ae_m, hidden_dim_m, cfg_m = load_modality_with_config(path, map_location=device)
        ae_m = ae_m.to(device)

        enc, dec = extract_encoder_decoder_from_pretrained(ae_m)
        encoders[mod] = enc
        decoders[mod] = dec
        hidden_dims[mod] = hidden_dim_m

        # pull modality-specific sentinel from config (defaults to 0.0 if older checkpoint)
        mask_values[mod] = cfg_m.get("mask_value", 0.0)

    # 2) Build dataset/loaders (aligned by common_samples)
    multi_ds = MultiOmicDataset({m: df.loc[common_samples] for m, df in multi_omic_data.items()})
    train_loader = get_dataloader(multi_ds, batch_size=batch_size, shuffle=shuffle_train, split_idx=train_idx)
    val_loader   = get_dataloader(multi_ds, batch_size=batch_size, shuffle=False,       split_idx=val_idx)
    test_loader  = get_dataloader(multi_ds, batch_size=batch_size, shuffle=False,       split_idx=test_idx) if test_idx is not None else None

    # 3) Assemble shared-space model
    model = MultiModalWithSharedSpace(
        encoders=encoders,
        decoders=decoders,
        hidden_dims=hidden_dims,
        shared_dim=shared_dim,
        proj_depth=proj_depth,
    ).to(device)

    # 4) (Optional) Freeze encoders/decoders
    if freeze_encoders_decoders:
        for p in model.encoders.parameters():
            p.requires_grad = False
        for p in model.decoders.parameters():
            p.requires_grad = False
        model.encoders.eval()
        model.decoders.eval()

    # 5) Optimizer
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # 6) Train/eval loops
    train_loss_hist = {"total": [], "recon": [], "contrast": [], "impute": []}
    val_loss_hist   = {"total": [], "recon": [], "contrast": [], "impute": []}
    
    for ep in range(1, epochs + 1):
        train_stats = finetune_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=opt,
            device=device,
            mask_values=mask_values,                    # NEW
            lambda_contrastive=lambda_contrast,
            lambda_recon=lambda_recon,
            lambda_impute=lambda_impute,
            modality_dropout_prob=modality_dropout_prob,
            feature_mask_p=feature_mask_p_train,
            alpha_mask_recon=alpha_mask_recon,          # NEW
            two_path_clean_for_contrast=two_path_clean_for_contrast,
        )

        val_stats = eval_finetune_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            mask_values=mask_values,                    # NEW
            lambda_contrastive=lambda_contrast,
            lambda_recon=lambda_recon,
            lambda_impute=lambda_impute,
            feature_mask_p=feature_mask_p_val,
            alpha_mask_recon=alpha_mask_recon,          # NEW
            two_path_clean_for_contrast=two_path_clean_for_contrast,
        )

        for k_src, k_dst in [
            ("total_loss","total"),
            ("recon_loss","recon"),
            ("contrast_loss","contrast"),
            ("impute_loss","impute")
        ]:
            train_loss_hist[k_dst].append(train_stats[k_src])
            val_loss_hist[k_dst].append(val_stats[k_src])

        if verbose:
            print(f"[Finetune] ep {ep:03d} | "
                  f"train tot {train_stats['total_loss']:.4f} | "
                  f"recon {train_stats['recon_loss']:.4f} | "
                  f"contr {train_stats['contrast_loss']:.4f} | "
                  f"impute {train_stats['impute_loss']:.4f}")
            print(f"                 |   val  tot {val_stats['total_loss']:.4f} | "
                  f"recon {val_stats['recon_loss']:.4f} | "
                  f"contr {val_stats['contrast_loss']:.4f} | "
                  f"impute {val_stats['impute_loss']:.4f}")

    return model, train_loss_hist, val_loss_hist, train_loader, val_loader, test_loader, opt

def save_shared_model(model, save_dir: str, epoch: int, train_loss_hist=None, val_loss_hist=None):
    """
    Save the full shared-space model (encoders, projections, decoders) and metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"shared_model_ep{epoch}.pt")

    # Save model state
    torch.save(model.state_dict(), model_path)

    # Optional: save training history as JSON
    if train_loss_hist is not None and val_loss_hist is not None:
        hist_path = os.path.join(save_dir, f"loss_history_ep{epoch}.json")
        with open(hist_path, "w") as f:
            json.dump({"train": train_loss_hist, "val": val_loss_hist}, f)

    print(f"[Saved] Shared model checkpoint → {model_path}")
    return model_path


def load_shared_model(
    model_class,
    encoders,
    decoders,
    hidden_dims,
    shared_dim: int,
    proj_depth: int,
    checkpoint_path: str,
    map_location=None
):
    """
    Rebuild a MultiModalWithSharedSpace instance and load its weights.
    """
    model = model_class(
        encoders=encoders,
        decoders=decoders,
        hidden_dims=hidden_dims,
        shared_dim=shared_dim,
        proj_depth=proj_depth,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
    model.eval()
    print(f"[Loaded] Shared model from {checkpoint_path}")
    return model