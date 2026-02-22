#!/usr/bin/env python3

# Odd non-breaking Windows issue where PyTorch's MKL/Intel OpenMP gets initialized during the save operation
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Minimal JEPA training pipeline for longitudinal patient sequences.
Architecture
------------
  token_embedding  : nn.Embedding(vocab_size, 64), mean-pooled per encounter
  context_encoder  : hand-rolled Transformer (2 layers, 2 heads, dim=64) → z_context
  target_encoder   : EMA copy of context_encoder (τ=0.996), no backprop → z_target
  predictor        : MLP(z_context ⊕ pos_emb → z_pred, hidden=128)
  loss             : MSE(z_pred, z_target)
"""

import argparse
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg") # Set rendering backend for saving plots w/o a display
import matplotlib.pyplot as plt

from src.config.io import MODEL_RUNS_BASE_DIR, PROCESSED_DIR, resolve_path


EMBED_DIM        = 64
NUM_HEADS        = 2
NUM_LAYERS       = 2
FFN_DIM          = 256      # 4 × EMBED_DIM
MAX_SEQ_LEN      = 128      # positional embedding capacity
PREDICTOR_HIDDEN = 128
TAU              = 0.996    # EMA momentum
LR               = 1e-3
BATCH_SIZE       = 8
PAD_IDX          = 0        # token index reserved for padding


# =============================================================================
# Vocab
# =============================================================================

def build_vocab(patients: list[dict]) -> dict[str, int]:
    """Map every unique ICD code and med name to a positive integer index.
    Index 0 is reserved for [PAD].
    """
    tokens: set[str] = set()
    for p in patients:
        for enc in p.get("encounters", []):
            tokens.update(enc.get("icd_codes", []))
            tokens.update(enc.get("meds", []))
    vocab: dict[str, int] = {"[PAD]": PAD_IDX}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i
    return vocab


# =============================================================================
# Dataset
# =============================================================================

def encode_encounter(enc: dict, vocab: dict[str, int]) -> list[int]:
    """Return a list of token indices for one encounter dict."""
    codes = enc.get("icd_codes", []) + enc.get("meds", [])
    ids = [vocab[c] for c in codes if c in vocab]
    return ids if ids else [PAD_IDX]   # at least one token


class JEPADataset(Dataset):
    """One sample per (patient, masked-encounter-index).

    For a patient with N encounters, N samples are created.  Each sample
    uses N-1 encounters as context and 1 as the prediction target.
    Patients with fewer than 2 encounters are skipped.
    """

    def __init__(self, patients: list[dict], vocab: dict[str, int]):
        self.samples: list[dict] = []
        for p in patients:
            encs = p.get("encounters", [])
            if len(encs) < 2:
                continue
            label  = int(p.get("label", 0))
            sid    = str(p["subject_id"])
            tokens = [encode_encounter(e, vocab) for e in encs]
            for mask_pos in range(len(encs)):
                self.samples.append({
                    "context":    [tokens[i] for i in range(len(encs)) if i != mask_pos],
                    "target":     tokens[mask_pos],
                    "mask_pos":   mask_pos,
                    "subject_id": sid,
                    "label":      label,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length encounter sequences and token lists for batching."""
    B = len(batch)

    # Determine maximum context length and maximum tokens-per-encounter
    max_ctx = max(len(item["context"]) for item in batch)
    all_enc_lens = [
        len(enc)
        for item in batch for enc in item["context"]
    ] + [len(item["target"]) for item in batch]
    max_tok = max(all_enc_lens) if all_enc_lens else 1

    # ctx_tokens[b, c, t] - token indices for context encounter c in batch b
    ctx_tokens   = torch.zeros(B, max_ctx, max_tok, dtype=torch.long)
    # ctx_tok_mask[b, c, t] - True where a real token exists (for mean-pool)
    ctx_tok_mask = torch.zeros(B, max_ctx, max_tok, dtype=torch.bool)
    # ctx_pad_mask[b, c] - True where encounter slot c is padding (for attn)
    ctx_pad_mask = torch.ones(B, max_ctx, dtype=torch.bool)

    tgt_tokens   = torch.zeros(B, max_tok, dtype=torch.long)
    tgt_tok_mask = torch.zeros(B, max_tok, dtype=torch.bool)

    for i, item in enumerate(batch):
        for j, enc in enumerate(item["context"]):
            n = len(enc)
            ctx_tokens[i, j, :n]   = torch.tensor(enc, dtype=torch.long)
            ctx_tok_mask[i, j, :n] = True
            ctx_pad_mask[i, j]     = False # this slot is a real encounter

        n = len(item["target"])
        tgt_tokens[i, :n]   = torch.tensor(item["target"], dtype=torch.long)
        tgt_tok_mask[i, :n] = True

    mask_pos   = torch.tensor([item["mask_pos"] for item in batch], dtype=torch.long)
    labels     = torch.tensor([item["label"]    for item in batch], dtype=torch.long)
    subject_ids = [item["subject_id"] for item in batch]

    return {
        "ctx_tokens":    ctx_tokens,     # (B, max_ctx, max_tok)
        "ctx_tok_mask":  ctx_tok_mask,   # (B, max_ctx, max_tok)
        "ctx_pad_mask":  ctx_pad_mask,   # (B, max_ctx)  True=padding slot
        "tgt_tokens":    tgt_tokens,     # (B, max_tok)
        "tgt_tok_mask":  tgt_tok_mask,   # (B, max_tok)
        "mask_pos":      mask_pos,       # (B,)
        "labels":        labels,         # (B,)
        "subject_ids":   subject_ids,
    }


# =============================================================================
# Embedding utility
# =============================================================================

def embed_and_pool(
    embedding: nn.Embedding,
    tokens: torch.Tensor,       # (..., max_tok)  LongTensor
    tok_mask: torch.Tensor,     # (..., max_tok)  BoolTensor  True=real
) -> torch.Tensor:              # (..., embed_dim)
    """Embed tokens then mean-pool over the token dimension, ignoring padding."""
    emb  = embedding(tokens)                # (..., max_tok, D)
    mask = tok_mask.float().unsqueeze(-1)   # (..., max_tok, 1)
    return (emb * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1.0)


# =============================================================================
# JEPA model
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

    Parameters
    ----------
    embed_dim : total model dimension (must be divisible by num_heads)
    num_heads : number of attention heads
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Simpler to implement as separate linear layers for q, k, v than a single big one
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        Q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, Dh).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # Mask out padded key positions, broadcast over (H, T_query)
            # (B, T) True=ignore
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B,1,1,T)
                float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)      # guard fully-padded rows

        out = torch.matmul(attn, V)                 # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer:
      x -> LN -> MHSA -> residual -> LN -> FFN (GELU) -> residual
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, 
                x: torch.Tensor, 
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """ Stack of N TransformerEncoderLayers + learned positional embedding """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        num_layers:  int,
        max_seq_len: int,
        ffn_dim:     int,
    ):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,                              # (B, T, D)
        key_padding_mask: torch.Tensor | None = None, # (B, T) True=padding
    ) -> torch.Tensor:                                # (B, D)
        """
        Returns:
            single sequence-level vector (mean-pool over valid positions), i.e. shape (B, embed_dim).
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.norm(x)

        # Mean-pool over valid (non-padded) positions
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)   # (B, T, 1)
            z = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            z = x.mean(dim=1)

        return z                                                 # (B, D)


class JEPA(nn.Module):
    """Joint Embedding Predictive Architecture for patient sequences.

    Components
    ----------
    token_embedding : shared Embedding table (ICD codes + med names)
    context_encoder : TransformerEncoder trained with gradient descent
    target_encoder  : EMA copy of context_encoder, no gradients
    mask_pos_emb    : learnable position embedding fed to predictor
    predictor       : MLP(z_context (outer) pos_emb) -> z_pred
    """

    def __init__(
        self,
        vocab_size:       int,
        embed_dim:        int = EMBED_DIM,
        num_heads:        int = NUM_HEADS,
        num_layers:       int = NUM_LAYERS,
        max_seq_len:      int = MAX_SEQ_LEN,
        ffn_dim:          int = FFN_DIM,
        predictor_hidden: int = PREDICTOR_HIDDEN,
        tau:              float = TAU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tau       = tau

        # Shared token embedding, used by context and target
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.context_encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers, max_seq_len, ffn_dim)

        # Target encoder: EMA shadow of context_encoder. No backprop.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Positional embedding for the mask (target) position
        self.mask_pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # Predictor: (z_context || pos_emb) -> z_pred
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, embed_dim),
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """Exponential moving average update of target encoder weights."""
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(self.tau).add_((1.0 - self.tau) * p_ctx.data)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_context : (B, D) context representation (differentiable)
        z_pred    : (B, D) predictor output       (differentiable)
        z_target  : (B, D) EMA target encoding    (no grad)
        """
        ctx_tokens   = batch["ctx_tokens"]    # (B, C, T_tok)
        ctx_tok_mask = batch["ctx_tok_mask"]  # (B, C, T_tok)
        ctx_pad_mask = batch["ctx_pad_mask"]  # (B, C)
        tgt_tokens   = batch["tgt_tokens"]    # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"]  # (B, T_tok)
        mask_pos     = batch["mask_pos"]      # (B,)

        B, C, T_tok = ctx_tokens.shape

        # -- Context path (with grads) --------------------------------
        # Embed and mean-pool each context encounter: (B, C, D)
        ctx_embs = embed_and_pool(
            self.token_embedding,
            ctx_tokens.view(B * C, T_tok),
            ctx_tok_mask.view(B * C, T_tok),
        ).view(B, C, self.embed_dim)

        z_context = self.context_encoder(ctx_embs, ctx_pad_mask)  # (B, D)

        # -- Target path (no grads - EMA encoder) ---------------------
        with torch.no_grad():
            # (B, 1, D) single encounter as target
            tgt_embs = embed_and_pool(self.token_embedding, tgt_tokens, tgt_tok_mask).unsqueeze(1)
            # single encounter: no padding, so pad mask is all False
            tgt_pad_mask = torch.zeros(B, 1, dtype=torch.bool, device=tgt_tokens.device)
            z_target = self.target_encoder(tgt_embs, tgt_pad_mask) # (B, D)

        # -- Predictor ----------------------------------------------------
        # Clamp mask_pos to stay within positional embedding capacity
        mask_pos_clamped = mask_pos.clamp(max=MAX_SEQ_LEN - 1)
        pos_emb = self.mask_pos_emb(mask_pos_clamped)                     # (B, D)
        z_pred  = self.predictor(torch.cat([z_context, pos_emb], dim=-1)) # (B, D)

        return z_context, z_pred, z_target


# =============================================================================
# Training loop
# =============================================================================

def train(
    model: JEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> list[float]:
    model.train()
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []

        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            _, z_pred, z_target = model(batch_dev)
            loss = F.mse_loss(z_pred, z_target)
            loss.backward()
            optimizer.step()

            model.update_target_encoder()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)
        print(f"  Epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}")

    return loss_history


# =============================================================================
# Embedding extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    model.eval()

    all_z_ctx:  list[np.ndarray] = []
    all_z_pred: list[np.ndarray] = []
    all_z_tgt:  list[np.ndarray] = []
    all_sids:   list[str]        = []
    all_mpos:   list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        z_ctx, z_pred, z_tgt = model(batch_dev)

        all_z_ctx.append(z_ctx.cpu().numpy())
        all_z_pred.append(z_pred.cpu().numpy())
        all_z_tgt.append(z_tgt.cpu().numpy())
        all_sids.extend(batch["subject_ids"])
        all_mpos.append(batch["mask_pos"].numpy())
        all_labels.append(batch["labels"].numpy())

    z_context     = np.concatenate(all_z_ctx,  axis=0)
    z_pred_arr    = np.concatenate(all_z_pred, axis=0)
    z_target      = np.concatenate(all_z_tgt,  axis=0)
    delta         = z_pred_arr - z_context
    subject_ids   = np.array(all_sids)
    mask_positions = np.concatenate(all_mpos,  axis=0)
    labels        = np.concatenate(all_labels, axis=0)

    return z_context, z_pred_arr, z_target, delta, subject_ids, mask_positions, labels


# =============================================================================
# Sanity checks
# =============================================================================

def run_sanity_checks(
    loss_history:  list[float],
    z_context:     np.ndarray,
    z_pred:        np.ndarray,
    z_target:      np.ndarray,
    delta:         np.ndarray,
) -> None:
    print("\n" + "=" * 62)
    print("SANITY CHECKS")
    print("=" * 62)

    results: list[tuple[bool, str]] = []

    # 1. Loss trends downward
    mid       = max(1, len(loss_history) // 2)
    first_avg = np.mean(loss_history[:mid])
    last_avg  = np.mean(loss_history[mid:])
    ok1       = bool(last_avg < first_avg)
    results.append((ok1, (
        f"Loss trends downward: "
        f"first-half avg={first_avg:.6f}, second-half avg={last_avg:.6f}"
    )))

    # 2. z_pred has non-trivial variance (std > 0.01 per dimension)
    pred_std          = z_pred.std(axis=0)
    frac_nontrivial   = float((pred_std > 0.01).mean())
    ok2               = frac_nontrivial > 0.5
    results.append((ok2, (
        f"z_pred non-trivial variance: "
        f"min_std={pred_std.min():.4f}, "
        f"frac_dims(>0.01)={frac_nontrivial:.1%}"
    )))

    # 3. ||Δ|| varies across patients
    delta_norms = np.linalg.norm(delta, axis=-1)
    delta_std   = float(delta_norms.std())
    ok3         = delta_std > 1e-4
    results.append((ok3, (
        f"||Δ|| varies across samples: "
        f"std={delta_std:.6f}, "
        f"min={delta_norms.min():.4f}, "
        f"max={delta_norms.max():.4f}"
    )))

    # 4. Predictor beats a random baseline of the same scale as z_target
    rng          = np.random.default_rng(42)
    rand_noise   = rng.standard_normal(z_pred.shape).astype(np.float32)
    rand_pred    = rand_noise * z_target.std() + z_target.mean()
    pred_dist    = float(np.linalg.norm(z_pred - z_target, axis=-1).mean())
    rand_dist    = float(np.linalg.norm(rand_pred - z_target, axis=-1).mean())
    ok4          = pred_dist < rand_dist
    results.append((ok4, (
        f"Predictor beats random: "
        f"||z_pred-z_tgt||={pred_dist:.4f} vs "
        f"||rand-z_tgt||={rand_dist:.4f}"
    )))

    # 5. All arrays have consistent N_samples
    n_samples = {
        "z_context":     z_context.shape[0],
        "z_pred":        z_pred.shape[0],
        "z_target":      z_target.shape[0],
        "delta":         delta.shape[0],
    }
    ok5 = len(set(n_samples.values())) == 1
    results.append((ok5, f"Consistent N_samples: {n_samples}"))

    for idx, (ok, msg) in enumerate(results, start=1):
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {idx}. {msg}")

    n_pass = sum(ok for ok, _ in results)
    print(f"\n  {n_pass}/{len(results)} checks passed.")
    print("=" * 62 + "\n")


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal JEPA training pipeline for patient sequences")
    parser.add_argument("--name",        default=f"model-{np.random.randint(1000, 9999)}", type=str, help="Name for output model folder")
    parser.add_argument("--data_dir",    default="", type=str, help="Path to sequences.jsonl folder")
    parser.add_argument("--output-dir",  default="", type=str, help="Directory for outputs")
    parser.add_argument("--n-patients",  default=50, type=int, help="Max patients to load")
    parser.add_argument("--epochs",      default=10, type=int)
    parser.add_argument("--seed",        default=42, type=int)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = resolve_path(args.data_dir, PROCESSED_DIR)
    out_dir = resolve_path(args.output_dir, dflt=MODEL_RUNS_BASE_DIR) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # -- Load patients -----------------------------------------------------
    print(f"Loading data from: {args.data_dir}")
    patients: list[dict] = []
    with open(data_dir / "sequences.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                patients.append(json.loads(line))

    patients = patients[: args.n_patients]
    print(f"Patients loaded:   {len(patients)}")

    # -- Build vocabulary --------------------------------------------------
    vocab = build_vocab(patients)
    print(f"Vocabulary size:   {len(vocab)}")

    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)
    print(f"Vocab saved →      {vocab_path}")

    # -- Build dataset -----------------------------------------------------
    dataset = JEPADataset(patients, vocab)
    print(f"Training samples:  {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(
            "No training samples produced.  "
            "Ensure that patients have at least 2 encounters."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # -- Build model -------------------------------------------------------
    model = JEPA(vocab_size=len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params:  {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -- Train -------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs (batch_size={BATCH_SIZE}, lr={LR}) …")
    loss_history = train(model, loader, optimizer, args.epochs, device)

    # -- Save checkpoint ---------------------------------------------------
    ckpt_path = out_dir / "model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size":       len(vocab),
            "embed_dim":        EMBED_DIM,
            "loss_history":     loss_history,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved → {ckpt_path}")

    # -- Loss curve --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("JEPA Training Loss")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    curve_path = out_dir / "loss_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved → {curve_path}")

    # -- Extract embeddings ------------------------------------------------
    print("\nExtracting embeddings …")
    extract_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    z_context, z_pred, z_target, delta, subject_ids, mask_positions, labels = \
        extract_embeddings(model, extract_loader, device)

    npz_path = out_dir / "embeddings.npz"
    np.savez(
        npz_path,
        z_context=z_context,
        z_pred=z_pred,
        z_target=z_target,
        delta=delta,
        subject_ids=subject_ids,
        mask_positions=mask_positions,
        labels=labels,
    )
    print(f"Embeddings saved → {npz_path}")
    print(f"  z_context     : {z_context.shape}  (N_samples, {EMBED_DIM})")
    print(f"  z_pred        : {z_pred.shape}")
    print(f"  z_target      : {z_target.shape}")
    print(f"  delta         : {delta.shape}")
    print(f"  subject_ids   : {subject_ids.shape}")
    print(f"  mask_positions: {mask_positions.shape}")
    print(f"  labels        : {labels.shape}")

    # -- Sanity checks -----------------------------------------------------
    run_sanity_checks(loss_history, z_context, z_pred, z_target, delta)


if __name__ == "__main__":
    main()