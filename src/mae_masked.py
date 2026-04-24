import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
from typing import Dict, List, Tuple
import pickle
from torch.optim import Adam
import pandas as pd

###############################################
# Utilities
###############################################

def build_mlp(dims: List[int], add_final_activation: bool=False, activation_dropout: float=0.0) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        is_last = (i == len(dims) - 2)
        if not is_last or add_final_activation:
            layers.append(nn.ReLU())
            if activation_dropout > 0.0:
                layers.append(nn.Dropout(p=activation_dropout))
    return nn.Sequential(*layers)

###############################################
# Phase 1: Pretrain *per-modality* Autoencoders
###############################################

class ModalityEncoder(nn.Module):
    """Encoder: input_dim -> ... -> hidden_dim (no shared projection here)."""
    def __init__(self, dims: List[int], activation_dropout: float=0.0):
        super().__init__()
        assert len(dims) >= 2, "dims must include [input_dim, ..., hidden_dim]"
        self.net = build_mlp(dims, add_final_activation=False, activation_dropout=activation_dropout)

    def forward(self, x):
        return self.net(x)


class TiedLinear(nn.Module):
    """
    Linear layer with weights tied to a source Linear's transpose.
    y = x @ W_src^T + b
    """
    def __init__(self, src_linear: nn.Linear, out_features: int):
        super().__init__()
        self.src = src_linear
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x.matmul(self.src.weight.t()) + self.bias


class TiedDecoder(nn.Module):
    """
    Not currently using
    Mirrors encoder Linear layers in reverse with tied weights.
    Assumes encoder is an nn.Sequential of Linear/ReLU/Dropout blocks from build_mlp.
    """
    def __init__(self, encoder: ModalityEncoder, input_dim: int, hidden_layers: List[int], activation_dropout: float=0.0):
        super().__init__()
        # collect encoder Linear layers in order
        enc_linears = [m for m in encoder.net if isinstance(m, nn.Linear)]
        # Build mirrored tied stack: hidden -> ... -> input
        layers = []
        # For all but the last tied layer, add ReLU (+ optional Dropout) after TiedLinear
        all_out_dims = list(reversed([l.in_features for l in enc_linears])) + [input_dim]
        # The number of tied steps equals number of encoder Linear layers
        for i, src_lin in enumerate(reversed(enc_linears)):
            out_dim = all_out_dims[i]
            layers.append(TiedLinear(src_lin, out_dim))
            is_last = (i == len(enc_linears) - 1)
            if not is_last:
                layers.append(nn.ReLU())
                if activation_dropout > 0.0:
                    layers.append(nn.Dropout(p=activation_dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class ModalityDecoder(nn.Module):
    """Decoder: hidden_dim -> ... -> input_dim (untied)."""
    def __init__(self, dims: List[int], activation_dropout: float=0.0):
        super().__init__()
        assert len(dims) >= 2, "dims must include [hidden_dim, ..., input_dim]"
        self.net = build_mlp(dims, add_final_activation=False, activation_dropout=activation_dropout)

    def forward(self, h):
        return self.net(h)



class ModalityAutoencoder(nn.Module):
    def __init__(self,
                 encoder_dims: List[int],
                 decoder_dims: List[int],
                 activation_dropout: float = 0.0,
                 denoising: bool = False,
                 mask_p: float = 0.0,
                 tied: bool = False,
                 hidden_layers: List[int] = None,
                 input_dim: int = None,
                 mask_value: float = 0.0,     # NEW: what we put in masked positions
                 loss_on_masked: bool = True # NEW: whether to only use masked positions in loss
                 ):
        super().__init__()
        self.denoising = denoising
        self.mask_p = mask_p
        self.tied = tied
        self.mask_value = mask_value
        self.loss_on_masked = loss_on_masked
        self._last_mask = None  # will store mask for the last forward pass

        self.encoder = ModalityEncoder(encoder_dims, activation_dropout)
        if tied:
            assert input_dim is not None and hidden_layers is not None, \
                "For tied decoder, pass input_dim and hidden_layers used by the encoder."
            self.decoder = TiedDecoder(
                self.encoder,
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                activation_dropout=activation_dropout,
            )
        else:
            self.decoder = ModalityDecoder(decoder_dims, activation_dropout)

    def _add_mask_noise(self, x):
        if not self.training or not self.denoising or self.mask_p <= 0.0:
            self._last_mask = torch.zeros_like(x, dtype=torch.bool)
            return x
        mask = torch.rand_like(x) < self.mask_p   # True where we mask
        x_noisy = x.clone()
        x_noisy[mask] = self.mask_value
        self._last_mask = mask
        return x_noisy

    def forward(self, x):
        x_in = self._add_mask_noise(x)
        h = self.encoder(x_in)
        x_recon = self.decoder(h)
        return h, x_recon

    
def pretrain_modality_epoch(
    ae: ModalityAutoencoder,
    dataloader: DataLoader,
    optimizer,
    device,
    l1_alpha: float = 0.0,
    alpha_mask: float = 1.0,  # weight for masked MSE in training loss
):
    """
    - Replaces true NaNs with ae.mask_value in the input.
    - Excludes true NaNs from BOTH overall and masked MSE.
    - Uses ae._last_mask (artificially masked entries) for masked loss.
    """
    ae.train()
    total_loss = 0.0
    total_overall = 0.0
    total_masked = 0.0
    n = 0

    for xb in dataloader:
        xb = xb.to(device)

        # 1) true-missing mask (NaNs in the raw data)
        orig_missing = torch.isnan(xb)

        # 2) replace NaNs with sentinel BEFORE forward
        xb_in = xb.clone()
        xb_in[orig_missing] = ae.mask_value

        optimizer.zero_grad()

        # forward (ModalityAutoencoder will additionally apply artificial masking)
        h, recon = ae(xb_in)

        diff_sq = (recon - xb_in) ** 2

        # 3) valid positions = not truly missing
        valid = ~orig_missing

        # Overall MSE: only over valid entries
        if valid.any():
            overall_mse = diff_sq[valid].mean()
        else:
            overall_mse = diff_sq.mean()

        # Masked MSE: artificially masked AND valid
        if ae.denoising and ae.loss_on_masked:
            mask_artificial = ae._last_mask.to(device)
            mask = mask_artificial & valid
            if mask.any():
                masked_mse = diff_sq[mask].mean()
            else:
                masked_mse = overall_mse
        else:
            masked_mse = overall_mse

        # Combined loss
        loss = alpha_mask * masked_mse + (1.0 - alpha_mask) * overall_mse

        if l1_alpha > 0:
            loss = loss + l1_alpha * h.abs().mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_overall += overall_mse.item()
        total_masked += masked_mse.item()
        n += 1

    avg_loss = total_loss / max(n, 1)
    avg_overall = total_overall / max(n, 1)
    avg_masked = total_masked / max(n, 1)
    return avg_loss, avg_overall, avg_masked




def eval_modality_epoch_masked(ae: ModalityAutoencoder, dataloader: DataLoader, device):
    """
    Eval with the same behavior:
    - Use ae.mask_value for true NaNs in the input.
    - Exclude true NaNs from both overall and masked MSE.
    Returns (overall_mse, masked_mse).
    """
    was_training = ae.training
    ae.train()  # enable internal masking so _last_mask is set
    total_overall, total_masked, n = 0.0, 0.0, 0

    with torch.no_grad():
        for xb in dataloader:
            xb = xb.to(device)
            orig_missing = torch.isnan(xb)
            xb_in = xb.clone()
            xb_in[orig_missing] = ae.mask_value

            _, recon = ae(xb_in)
            diff_sq = (recon - xb_in) ** 2

            valid = ~orig_missing
            if valid.any():
                overall_mse = diff_sq[valid].mean()
            else:
                overall_mse = diff_sq.mean()

            mask_artificial = ae._last_mask.to(device)
            mask = mask_artificial & valid
            if mask.any():
                masked_mse = diff_sq[mask].mean()
            else:
                masked_mse = overall_mse

            total_overall += overall_mse.item()
            total_masked += masked_mse.item()
            n += 1

    if not was_training:
        ae.eval()

    return total_overall / max(n, 1), total_masked / max(n, 1)





###############################################
# Phase 2: Add projection -> shared space -> reverse_projection and finetune
###############################################

class ProjectionHead(nn.Module):
    """ hidden_dim -> shared_dim (optionally with a small MLP) """
    def __init__(self, hidden_dim: int, shared_dim: int, depth: int=1, activation_dropout: float=0.0):
        super().__init__()
        if depth == 1:
            self.proj = nn.Linear(hidden_dim, shared_dim)
        else:
            dims = [hidden_dim] + [hidden_dim] * (depth - 2) + [shared_dim]
            self.proj = build_mlp(dims, add_final_activation=False, activation_dropout=activation_dropout)

    def forward(self, h):
        return self.proj(h)


class ReverseProjectionHead(nn.Module):
    """ shared_dim -> hidden_dim (mirrors ProjectionHead) """
    def __init__(self, shared_dim: int, hidden_dim: int, depth: int=1, activation_dropout: float=0.0):
        super().__init__()
        if depth == 1:
            self.rproj = nn.Linear(shared_dim, hidden_dim)
        else:
            dims = [shared_dim] + [shared_dim] * (depth - 2) + [hidden_dim]
            self.rproj = build_mlp(dims, add_final_activation=False, activation_dropout=activation_dropout)

    def forward(self, z):
        return self.rproj(z)


class MultiModalWithSharedSpace(nn.Module):
    """
    Wraps pretrained per-modality autoencoders and inserts
    hidden -> projection -> shared -> reverse_projection -> hidden -> decoder

    During finetuning, we:
      - compute recon loss on the outputs
      - compute contrastive loss on the shared embeddings across modalities
    """
    def __init__(self,
                 encoders: Dict[str, ModalityEncoder],
                 decoders: Dict[str, ModalityDecoder],
                 hidden_dims: Dict[str, int],
                 shared_dim: int,
                 proj_depth: int=1,
                 activation_dropout: float=0.0):
        super().__init__()
        self.modalities = list(encoders.keys())
        self.shared_dim = shared_dim

        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

        self.projections = nn.ModuleDict({m: ProjectionHead(hidden_dims[m], shared_dim, proj_depth, activation_dropout)
                                          for m in self.modalities})
        self.rev_projections = nn.ModuleDict({m: ReverseProjectionHead(shared_dim, hidden_dims[m], proj_depth, activation_dropout)
                                              for m in self.modalities})

    def forward(self, batch: Dict[str, torch.Tensor]):
        shared_embeddings = {}
        reconstructions = {}
        hidden_states = {}

        for mod, xb in batch.items():
            h = self.encoders[mod](xb)
            z = self.projections[mod](h)              # to shared space
            h_hat = self.rev_projections[mod](z)      # back to hidden
            x_recon = self.decoders[mod](h_hat)

            hidden_states[mod] = h
            shared_embeddings[mod] = z
            reconstructions[mod] = x_recon

        return shared_embeddings, reconstructions, hidden_states
    
    
###############################################
# Losses and helpers
###############################################

def apply_modality_dropout(batch: Dict[str, torch.Tensor], dropout_prob: float=0.2):
    kept = {}
    for mod, x in batch.items():
        if random.random() > dropout_prob:
            kept[mod] = x
    return kept if kept else batch

def apply_feature_mask_noise(batch: Dict[str, torch.Tensor], p: float = 0.1):
    """Multiply features by a Bernoulli(1-p) keep-mask, per-modality."""
    if p <= 0.0:
        return batch
    out = {}
    for mod, x in batch.items():
        m = (torch.rand_like(x) > p).float()
        out[mod] = x * m
    return out


def reconstruction_loss_with_masks(
    target_batch: Dict[str, torch.Tensor],
    reconstructions: Dict[str, torch.Tensor],
    orig_missing_masks: Dict[str, torch.Tensor],
    artificial_masks: Dict[str, torch.Tensor] = None,
    alpha_mask: float = 0.5,
):
    """
    For each modality:
      - overall_mse: MSE over non-NaN entries
      - masked_mse: MSE over entries artificially masked (and non-NaN)
      - loss_mod = alpha_mask * masked_mse + (1 - alpha_mask) * overall_mse

    If artificial_masks is None, or p=0, this reduces to overall_mse.
    """
    per_mod_loss = {}
    total = 0.0

    for mod, target in target_batch.items():
        recon = reconstructions[mod]
        diff_sq = (recon - target) ** 2

        valid = ~orig_missing_masks[mod]
        if valid.any():
            overall_mse = diff_sq[valid].mean()
        else:
            overall_mse = diff_sq.mean()

        if artificial_masks is not None:
            mask = artificial_masks[mod] & valid
            if mask.any():
                masked_mse = diff_sq[mask].mean()
            else:
                masked_mse = overall_mse
            loss_mod = alpha_mask * masked_mse + (1.0 - alpha_mask) * overall_mse
        else:
            loss_mod = overall_mse

        per_mod_loss[mod] = loss_mod
        total = total + loss_mod

    return total, per_mod_loss


def contrastive_loss(embeddings: Dict[str, torch.Tensor], temperature: float=0.1):
    mods = list(embeddings.keys())
    if len(mods) < 2:
        any_tensor = next(iter(embeddings.values()))
        return any_tensor.new_tensor(0.0)

    loss = 0.0
    count = 0
    for i in range(len(mods)):
        for j in range(i+1, len(mods)):
            z1 = embeddings[mods[i]]
            z2 = embeddings[mods[j]]
            sim_ab = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / temperature
            labels = torch.arange(z1.size(0), device=z1.device)
            loss += F.cross_entropy(sim_ab, labels)

            sim_ba = F.cosine_similarity(z2.unsqueeze(1), z1.unsqueeze(0), dim=-1) / temperature
            loss += F.cross_entropy(sim_ba, labels)
            count += 2
    return loss / max(count, 1)


# Imputation loss = reconstruct each target modality from the mean shared embedding of *other* modalities
def imputation_loss(
    target_batch: Dict[str, torch.Tensor],
    embeddings: Dict[str, torch.Tensor],
    model: MultiModalWithSharedSpace,
    orig_missing_masks: Dict[str, torch.Tensor],
):
    """
    Same idea as before, but:
      - target_batch has NaNs replaced by sentinel
      - loss excludes original NaNs
    """
    per_mod = {}
    losses = []

    for target in embeddings.keys():
        if target not in target_batch:
            continue

        other = [m for m in embeddings.keys() if m != target and m in target_batch]
        if len(other) == 0:
            continue

        z_mean = torch.stack([embeddings[m] for m in other], dim=0).mean(dim=0)
        h_hat_imp = model.rev_projections[target](z_mean)
        x_imp = model.decoders[target](h_hat_imp)

        target_t = target_batch[target]
        diff_sq = (x_imp - target_t) ** 2
        valid = ~orig_missing_masks[target]

        if valid.any():
            loss_t = diff_sq[valid].mean()
        else:
            loss_t = diff_sq.mean()

        per_mod[target] = loss_t
        losses.append(loss_t)

    if len(losses) == 0:
        any_tensor = next(iter(embeddings.values()))
        return any_tensor.new_tensor(0.0), per_mod

    return torch.stack(losses).mean(), per_mod



###############################################
# Finetuning loops (shared-space model)
###############################################

def finetune_epoch(
    model: MultiModalWithSharedSpace,
    dataloader: DataLoader,
    optimizer,
    device,
    mask_values: Dict[str, float],
    lambda_contrastive: float = 1.0,
    lambda_recon: float = 1.0,
    lambda_impute: float = 1.0,
    modality_dropout_prob: float = 0.2,
    feature_mask_p: float = 0.1,
    alpha_mask_recon: float = 0.5,
    two_path_clean_for_contrast: bool = False,
):
    model.train()
    sums = {"total":0.0, "recon":0.0, "contrast":0.0, "impute":0.0}
    per_mod_recon_sums, per_mod_impute_sums = {}, {}
    n = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = apply_modality_dropout(batch, modality_dropout_prob)

        # Step 1: NaNs -> sentinel
        batch_clean, orig_missing_masks = prepare_clean_batch(batch, mask_values)

        if two_path_clean_for_contrast:
            # clean path
            shared_clean, _, _ = model(batch_clean)

            # noisy path: sentinel masking
            noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                batch_clean, mask_values, feature_mask_p
            )
            _, recons_noisy, _ = model(noisy_batch)

            # Reconstruction loss (overall + masked)
            rloss, per_mod_recon = reconstruction_loss_with_masks(
                target_batch=batch_clean,
                reconstructions=recons_noisy,
                orig_missing_masks=orig_missing_masks,
                artificial_masks=artificial_masks,
                alpha_mask=alpha_mask_recon,
            )

            # Contrastive on clean embeddings
            closs = contrastive_loss(shared_clean)

            # Imputation from clean embeddings, ignoring true NaNs
            iloss, per_mod_impute = imputation_loss(
                target_batch=batch_clean,
                embeddings=shared_clean,
                model=model,
                orig_missing_masks=orig_missing_masks,
            )

        else:
            # single-path: only noisy inputs
            noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                batch_clean, mask_values, feature_mask_p
            )
            shared, recons, _ = model(noisy_batch)

            rloss, per_mod_recon = reconstruction_loss_with_masks(
                target_batch=batch_clean,
                reconstructions=recons,
                orig_missing_masks=orig_missing_masks,
                artificial_masks=artificial_masks,
                alpha_mask=alpha_mask_recon,
            )

            closs = contrastive_loss(shared)

            iloss, per_mod_impute = imputation_loss(
                target_batch=batch_clean,
                embeddings=shared,
                model=model,
                orig_missing_masks=orig_missing_masks,
            )

        total = lambda_recon* rloss + lambda_contrastive * closs + lambda_impute * iloss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        sums["total"]    += total.item()
        sums["recon"]    += rloss.item()
        sums["contrast"] += closs.item()
        sums["impute"]   += iloss.item()

        for m, v in per_mod_recon.items():
            per_mod_recon_sums[m] = per_mod_recon_sums.get(m, 0.0) + v.item()
        for m, v in per_mod_impute.items():
            per_mod_impute_sums[m] = per_mod_impute_sums.get(m, 0.0) + v.item()
        n += 1

    avg = {
        "total_loss":   sums["total"]/max(n,1),
        "recon_loss":   sums["recon"]/max(n,1),
        "contrast_loss":sums["contrast"]/max(n,1),
        "impute_loss":  sums["impute"]/max(n,1),
        "modality_losses": {
            "recon":  {m: v/max(n,1) for m, v in per_mod_recon_sums.items()},
            "impute": {m: v/max(n,1) for m, v in per_mod_impute_sums.items()},
        }
    }
    return avg



def eval_finetune_epoch(
    model: MultiModalWithSharedSpace,
    dataloader: DataLoader,
    device,
    mask_values: Dict[str, float],
    lambda_contrastive: float = 1.0,
    lambda_recon: float = 1.0,
    lambda_impute: float = 1.0,
    feature_mask_p: float = 0.0,
    alpha_mask_recon: float = 0.5,
    two_path_clean_for_contrast: bool = False,
):
    model.eval()
    sums = {"total":0.0, "recon":0.0, "contrast":0.0, "impute":0.0}
    per_mod_recon_sums, per_mod_impute_sums = {}, {}
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            batch_clean, orig_missing_masks = prepare_clean_batch(batch, mask_values)

            if two_path_clean_for_contrast:
                shared_clean, _, _ = model(batch_clean)
                noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                    batch_clean, mask_values, feature_mask_p
                )
                _, recons_noisy, _ = model(noisy_batch)

                rloss, per_mod_recon = reconstruction_loss_with_masks(
                    target_batch=batch_clean,
                    reconstructions=recons_noisy,
                    orig_missing_masks=orig_missing_masks,
                    artificial_masks=artificial_masks,
                    alpha_mask=alpha_mask_recon,
                )
                closs = contrastive_loss(shared_clean)
                iloss, per_mod_impute = imputation_loss(
                    target_batch=batch_clean,
                    embeddings=shared_clean,
                    model=model,
                    orig_missing_masks=orig_missing_masks,
                )
            else:
                noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                    batch_clean, mask_values, feature_mask_p
                )
                shared, recons, _ = model(noisy_batch)

                rloss, per_mod_recon = reconstruction_loss_with_masks(
                    target_batch=batch_clean,
                    reconstructions=recons,
                    orig_missing_masks=orig_missing_masks,
                    artificial_masks=artificial_masks,
                    alpha_mask=alpha_mask_recon,
                )
                closs = contrastive_loss(shared)
                iloss, per_mod_impute = imputation_loss(
                    target_batch=batch_clean,
                    embeddings=shared,
                    model=model,
                    orig_missing_masks=orig_missing_masks,
                )

            total = lambda_recon * rloss + lambda_contrastive * closs + lambda_impute * iloss

            sums["total"]    += total.item()
            sums["recon"]    += rloss.item()
            sums["contrast"] += closs.item()
            sums["impute"]   += iloss.item()

            for m, v in per_mod_recon.items():
                per_mod_recon_sums[m] = per_mod_recon_sums.get(m, 0.0) + v.item()
            for m, v in per_mod_impute.items():
                per_mod_impute_sums[m] = per_mod_impute_sums.get(m, 0.0) + v.item()
            n += 1

    avg = {
        "total_loss":   sums["total"]/max(n,1),
        "recon_loss":   sums["recon"]/max(n,1),
        "contrast_loss":sums["contrast"]/max(n,1),
        "impute_loss":  sums["impute"]/max(n,1),
        "modality_losses": {
            "recon":  {m: v/max(n,1) for m, v in per_mod_recon_sums.items()},
            "impute": {m: v/max(n,1) for m, v in per_mod_impute_sums.items()},
        }
    }
    return avg


###############################################
# Helper builders & checkpoints
###############################################

def build_pretrain_ae_for_modality(
    input_dim: int,
    hidden_layers: List[int],
    activation_dropout: float = 0.0,
    denoising: bool = False,
    mask_p: float = 0.0,
    tied: bool = False,
    mask_value: float = 0.0,
    loss_on_masked: bool = True,
) -> Tuple[ModalityAutoencoder, int]:
    """
    Returns (autoencoder, hidden_dim). Example:
      input_dim=225, hidden_layers=[112, 56, 128] -> hidden_dim=128
    """
    hidden_dim = hidden_layers[-1]

    enc_dims = [input_dim] + hidden_layers
    dec_dims = [hidden_dim] + list(reversed(hidden_layers[:-1])) + [input_dim]

    ae = ModalityAutoencoder(
        encoder_dims=enc_dims,
        decoder_dims=dec_dims,
        activation_dropout=activation_dropout,
        denoising=denoising,
        mask_p=mask_p,
        tied=tied,
        hidden_layers=hidden_layers,
        input_dim=input_dim,
        mask_value=mask_value,
        loss_on_masked=loss_on_masked,
    )
    return ae, hidden_dim


def extract_encoder_decoder_from_pretrained(ae: ModalityAutoencoder) -> Tuple[ModalityEncoder, ModalityDecoder]:
    enc = ae.encoder
    dec = ae.decoder
    return enc, dec


def save_modality_with_config(ae: ModalityAutoencoder, config: dict, path_prefix: str):
    save_data = {
        'state_dict': ae.state_dict(),
        'config': config
    }
    torch.save(save_data, f"{path_prefix}.pt")

def load_modality_with_config(path: str, map_location=None):
    data = torch.load(path, map_location=map_location)
    config = data['config']

    ae, hidden_dim = build_pretrain_ae_for_modality(
        config['input_dim'],
        config['hidden_layers'],
        activation_dropout=config.get('activation_dropout', 0.0),
        denoising=config.get('denoising', False),
        mask_p=config.get('mask_p', 0.0),
        tied=config.get('tied', False),
        mask_value=config.get('mask_value', 0.0),          # NEW
        loss_on_masked=config.get('loss_on_masked', True)  # NEW
    )
    ae.load_state_dict(data['state_dict'])
    return ae, hidden_dim, config

def prepare_clean_batch(batch: Dict[str, torch.Tensor],
                        mask_values: Dict[str, float]):
    """
    Replace true NaNs with modality-specific sentinel values.
    Returns:
      - batch_clean: dict[mod] -> tensor with NaNs replaced
      - orig_missing_masks: dict[mod] -> bool tensor for true NaNs
    """
    batch_clean = {}
    orig_missing_masks = {}
    for mod, x in batch.items():
        orig_missing = torch.isnan(x)
        x_clean = x.clone()
        x_clean[orig_missing] = mask_values.get(mod, 0.0)
        batch_clean[mod] = x_clean
        orig_missing_masks[mod] = orig_missing
    return batch_clean, orig_missing_masks

def apply_feature_mask_noise_with_sentinels(
    batch_clean: Dict[str, torch.Tensor],
    mask_values: Dict[str, float],
    p: float = 0.1
):
    """
    For each modality, with prob p per feature, replace with that modality’s sentinel.

    Returns:
      noisy_batch: dict[mod] -> tensor (corrupted input)
      artificial_masks: dict[mod] -> bool tensor (True where we masked)
    """
    noisy_batch = {}
    artificial_masks = {}
    if p <= 0.0:
        for mod, x in batch_clean.items():
            noisy_batch[mod] = x
            artificial_masks[mod] = torch.zeros_like(x, dtype=torch.bool)
        return noisy_batch, artificial_masks

    for mod, x in batch_clean.items():
        m = torch.rand_like(x) < p  # True where we mask
        x_noisy = x.clone()
        x_noisy[m] = mask_values.get(mod, 0.0)
        noisy_batch[mod] = x_noisy
        artificial_masks[mod] = m
    return noisy_batch, artificial_masks
