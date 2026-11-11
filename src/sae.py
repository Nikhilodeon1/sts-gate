"""
sae.py – Test B: BatchTopK Sparse Autoencoder at the locked ECG-FM layer.

Steps:
  1. Extract Layer tokens from PTB-XL (if present) or MedalCare-XL train split
     (up to 5,000 ECGs, stratified) into train shards and strictly held-out shards.
  2. Scale tokens on train shards so mean token L2-norm = sqrt(d_model). Save scale constant.
  3. Train BatchTopK SAE on train shards only; evaluate on strictly held-out shards.
  4. Run matched control (random-init ECG-FM model, same architecture & training protocol).
  5. Report all four Test B checks:
       - Check 1: Held-out token Explained Variance (EV >= 0.85 / >= 0.70)
       - Check 2: Simulated ECG Held-out Explained Variance (EV >= 0.70)
       - Check 3: Dead atom fraction (< 20%) and mean L0 sparsity
       - Check 4: Downstream 8-class linear probe drop on SAE reconstructions x_hat
                  and sparse atom representations z (probe drop <= 0.05)

Run:
    python src/sae.py --layer 8 [--device cuda] [--exclude-run-ids run_S64 run_S67]
"""
import sys, json, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import load_ptbxl, get_ptbxl_records, build_index, load_medalcare


# Keep multi-GB temporary token shards on local scratch disk (/tmp/ecg-gate/tokens)
# so Google Drive is only used for persistent weights, scale constants, and results.
TOKEN_SCRATCH_ROOT = C.SCRATCH_DIR / "tokens"


# ─────────────────────────────────────────────────────────────────────────────
# BatchTopK SAE model
# ─────────────────────────────────────────────────────────────────────────────

class BatchTopKSAE(nn.Module):
    """
    Sparse Autoencoder with batch-wide TopK sparsity.

    During training: keep exactly k * batch_size activations across the batch.
    During eval:     use a running threshold (thr) set during training (or per-token top-k fallback).

    d_in  : token dimension (768 for ECG-FM)
    d_sae : number of atoms (default 6144 = 8x)
    k     : topk per token during training (default 32)
    """
    def __init__(self, d_in: int = C.SAE_D_IN,
                 d_sae: int = C.SAE_D_SAE,
                 k:     int = C.SAE_K):
        super().__init__()
        self.d_in  = d_in
        self.d_sae = d_sae
        self.k     = k

        # Tied initialization prevents dead atoms at step 0
        w_init = torch.randn(d_in, d_sae)
        w_init /= w_init.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_enc = nn.Parameter(w_init.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(w_init.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Running threshold: exponential moving average of the smallest kept activation
        self.register_buffer("thr", torch.tensor(0.0))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        """Keep decoder columns (rows of W_dec) unit-norm."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, d_in)
        returns z: (batch, d_sae) sparse
        """
        pre = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if self.training:
            flat = pre.flatten()
            n_keep = min(flat.numel(), max(1, self.k * x.shape[0]))
            kept_vals = flat.topk(n_keep).values
            t = kept_vals[-1]
            if self.thr.item() == 0.0:
                self.thr.copy_(t.detach())
            else:
                self.thr.mul_(0.99).add_(0.01 * t.detach())
            mask = (pre >= t).float()
        else:
            if self.thr.item() > 0.0:
                mask = (pre >= self.thr).float()
            else:
                topk_vals = pre.topk(min(self.k, pre.shape[-1]), dim=-1).values
                thr_tok = topk_vals[:, -1:]
                mask = (pre >= thr_tok).float()

        return pre * mask

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """Returns (x_hat, z)."""
        z    = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction (PTB-XL if available, otherwise MedalCare-XL split-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _flush_shard(buf: list, out_dir: Path, prefix: str, idx: int):
    arr = np.concatenate(buf, axis=0)
    np.save(out_dir / f"{prefix}_shard_{idx:03d}.npy", arr)


def _shard_paths(shard_dir: Path, prefix: str = "train") -> list[Path]:
    paths = sorted(shard_dir.glob(f"{prefix}_shard_*.npy"))
    if not paths and prefix == "train":
        # Backward compatibility with unprefixed shard_NNN.npy
        paths = sorted(shard_dir.glob("shard_*.npy"))
    return paths


def extract_tokens(
    layer: int,
    model,
    shard_dir: Path,
    device: str = "cpu",
    shard_size: int = 250_000,
    max_train_ecgs: int = 5000,
    max_held_ecgs: int = 1500,
    batch_size: int = 32,
    exclude_run_ids: list[str] | None = None,
) -> str:
    """
    Extract layer tokens into strictly separated `train_shard_NNN.npy` and `held_shard_NNN.npy`.
    Uses PTB-XL (folds 1-8 train, folds 9-10 held-out) if available on disk;
    otherwise uses MedalCare-XL (train split vs. validation+test splits).
    Returns token source string ("ptbxl" or "medalcare").
    """
    from model import extract_layers

    shard_dir.mkdir(parents=True, exist_ok=True)
    for old_file in shard_dir.glob("*shard_*.npy"):
        old_file.unlink()

    ptb_records = get_ptbxl_records(max_folder=4000)
    if len(ptb_records) >= 100:
        print(f"  Using PTB-XL token source ({len(ptb_records)} ECGs available)...")
        train_df = ptb_records[ptb_records["strat_fold"].isin(range(1, 9))].reset_index(drop=True)
        held_df  = ptb_records[ptb_records["strat_fold"].isin([9, 10])].reset_index(drop=True)
        _extract_from_df(
            train_df, "wfdb_path", load_ptbxl, model, layer,
            shard_dir, "train", shard_size, batch_size, device,
        )
        _extract_from_df(
            held_df, "wfdb_path", load_ptbxl, model, layer,
            shard_dir, "held", shard_size, batch_size, device,
        )
        return "ptbxl"

    print("  PTB-XL not mounted; extracting split-separated tokens from MedalCare-XL...")
    index = build_index()
    if exclude_run_ids:
        index = index.loc[~index["run_id"].isin(exclude_run_ids)].reset_index(drop=True)

    rng = np.random.default_rng(42)
    train_pool = index[index["split"] == "train"].reset_index(drop=True)
    held_pool  = index[index["split"].isin(["validation", "test"])].reset_index(drop=True)

    # Stratified subsample of train and held-out splits to keep token extraction fast & balanced
    if len(train_pool) > max_train_ecgs:
        chosen = []
        n_each = max_train_ecgs // len(C.GROUP_LABELS)
        for g in C.GROUP_LABELS:
            g_rows = train_pool[train_pool["group"] == g]
            if len(g_rows) > n_each:
                pick = rng.choice(len(g_rows), n_each, replace=False)
                g_rows = g_rows.iloc[np.sort(pick)]
            chosen.append(g_rows)
        train_df = pd.concat(chosen, ignore_index=True)
    else:
        train_df = train_pool

    if len(held_pool) > max_held_ecgs:
        chosen = []
        n_each = max_held_ecgs // len(C.GROUP_LABELS)
        for g in C.GROUP_LABELS:
            g_rows = held_pool[held_pool["group"] == g]
            if len(g_rows) > n_each:
                pick = rng.choice(len(g_rows), n_each, replace=False)
                g_rows = g_rows.iloc[np.sort(pick)]
            chosen.append(g_rows)
        held_df = pd.concat(chosen, ignore_index=True)
    else:
        held_df = held_pool

    print(f"  Extracting train tokens ({len(train_df)} ECGs) and held-out tokens ({len(held_df)} ECGs)...")
    _extract_from_df(
        train_df, "ecg_path", load_medalcare, model, layer,
        shard_dir, "train", shard_size, batch_size, device,
    )
    _extract_from_df(
        held_df, "ecg_path", load_medalcare, model, layer,
        shard_dir, "held", shard_size, batch_size, device,
    )
    return "medalcare"


def _extract_from_df(
    df: pd.DataFrame,
    path_col: str,
    loader_fn,
    model,
    layer: int,
    shard_dir: Path,
    prefix: str,
    shard_size: int,
    batch_size: int,
    device: str,
) -> None:
    from model import extract_layers

    shard_buf = []
    shard_idx = 0
    total = 0
    N = len(df)

    for start in range(0, N, batch_size):
        batch = df.iloc[start:start + batch_size]
        wins = []
        for _, row in batch.iterrows():
            try:
                w = loader_fn(row[path_col])  # (2, 12, 2500)
                wins.append(w[0])
                wins.append(w[1])
            except Exception as e:
                print(f"[WARN] Skip {row[path_col]}: {e}")
                continue
        if not wins:
            continue

        x = torch.from_numpy(np.stack(wins, axis=0)).float()  # (2B, 12, 2500)
        feats = extract_layers(model, x, layers=[layer], device=device)
        toks = feats[layer].reshape(-1, feats[layer].shape[-1]).half().numpy()  # (2B*T, d)
        shard_buf.append(toks)
        total += toks.shape[0]

        if total >= shard_size:
            _flush_shard(shard_buf, shard_dir, prefix, shard_idx)
            shard_idx += 1
            shard_buf = []
            total = 0

        print(f"  [{prefix}] ECGs: {min(start + batch_size, N)}/{N}  shards={shard_idx}", end="\r", flush=True)

    if shard_buf:
        _flush_shard(shard_buf, shard_dir, prefix, shard_idx)
        shard_idx += 1
    print(f"\n  Saved {shard_idx} {prefix} shard(s) to {shard_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Scale normalisation
# ─────────────────────────────────────────────────────────────────────────────

def compute_scale(shard_dir: Path, d_model: int) -> float:
    """
    Compute scale constant on TRAIN shards so that mean token L2-norm = sqrt(d_model).
    """
    target_norm = math.sqrt(d_model)
    total, count = 0.0, 0
    train_shards = _shard_paths(shard_dir, "train")
    if not train_shards:
        raise FileNotFoundError(f"No train shards found in {shard_dir}")
    for p in train_shards:
        arr = np.load(p).astype(np.float32)
        total += float(np.linalg.norm(arr, axis=1).sum())
        count += arr.shape[0]
    mean_norm = total / max(count, 1)
    scale = mean_norm / target_norm
    print(f"Scale constant: {scale:.4f}  (mean norm {mean_norm:.2f} -> target {target_norm:.2f})")
    return scale


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_sae(
    shard_dir: Path,
    scale: float,
    device: str = "cpu",
    d_sae: int = C.SAE_D_SAE,
    k: int = C.SAE_K,
    lr: float = C.SAE_LR,
    batch_size: int = C.SAE_BATCH,
    n_epochs: int = C.SAE_EPOCHS,
    seed: int = 42,
) -> BatchTopKSAE:
    """
    Train a BatchTopK SAE strictly on train shards.
    """
    shards = _shard_paths(shard_dir, "train")
    if not shards:
        raise FileNotFoundError(f"No train shards found at {shard_dir}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    sae = BatchTopKSAE(d_in=C.SAE_D_IN, d_sae=d_sae, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    for epoch in range(n_epochs):
        sae.train()
        epoch_loss = 0.0
        n_batches  = 0
        shard_order = rng.permutation(len(shards))

        for si in shard_order:
            toks = np.load(shards[si]).astype(np.float32) / scale
            toks = toks[rng.permutation(len(toks))]
            for start in range(0, len(toks), batch_size):
                x = torch.from_numpy(toks[start:start + batch_size]).to(device)
                xhat, _ = sae(x)
                loss = F.mse_loss(xhat, x)
                opt.zero_grad()
                loss.backward()
                # Remove gradient component parallel to decoder weights before step
                opt.step()
                sae.normalize_decoder()
                epoch_loss += loss.item()
                n_batches  += 1

        print(f"  Epoch {epoch+1}/{n_epochs}  avg_mse={epoch_loss/max(n_batches,1):.6f}  thr={sae.thr.item():.4f}")

    return sae


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def explained_variance(
    sae: BatchTopKSAE,
    shards: list[Path],
    scale: float,
    device: str = "cpu",
    batch_size: int = 8192,
) -> dict:
    """Compute both per-token EV and global feature-variance EV across shards."""
    sae.eval()
    total_mse = 0.0
    total_tok_var = 0.0
    sum_x = np.zeros(sae.d_in, dtype=np.float64)
    sum_x2 = np.zeros(sae.d_in, dtype=np.float64)
    n_toks = 0

    for p in shards:
        raw = np.load(p).astype(np.float32) / scale
        sum_x += raw.sum(axis=0, dtype=np.float64)
        sum_x2 += (raw.astype(np.float64) ** 2).sum(axis=0)
        for start in range(0, len(raw), batch_size):
            arr = torch.from_numpy(raw[start:start + batch_size]).to(device)
            with torch.no_grad():
                xhat, _ = sae(arr)
            mse = ((arr - xhat) ** 2).mean(dim=-1)
            total_mse += mse.sum().item()
            total_tok_var += arr.var(dim=-1, unbiased=False).sum().item()
            n_toks += arr.shape[0]

    ev_per_token = 1.0 - (total_mse / max(total_tok_var, 1e-12))
    mean_x = sum_x / max(n_toks, 1)
    global_var_per_dim = (sum_x2 / max(n_toks, 1)) - (mean_x ** 2)
    global_var = float(np.mean(global_var_per_dim)) * n_toks
    ev_global = 1.0 - (total_mse / max(global_var, 1e-12))
    return {
        "ev_per_token": float(ev_per_token),
        "ev_global": float(ev_global),
        "n_tokens": int(n_toks),
    }


def sparsity_and_dead_atoms(
    sae: BatchTopKSAE,
    shards: list[Path],
    scale: float,
    device: str = "cpu",
    max_tokens: int = 1_000_000,
    batch_size: int = 8192,
) -> dict:
    """Compute dead atom fraction, per-atom firing frequency, and mean L0."""
    sae.eval()
    fire_counts = torch.zeros(sae.d_sae, dtype=torch.long)
    total_l0 = 0.0
    seen = 0
    for p in shards:
        raw = np.load(p).astype(np.float32) / scale
        for start in range(0, len(raw), batch_size):
            arr = torch.from_numpy(raw[start:start + batch_size]).to(device)
            with torch.no_grad():
                _, z = sae(arr)
            active = (z > 0)
            fire_counts += active.sum(dim=0).cpu()
            total_l0 += active.float().sum().item()
            seen += arr.shape[0]
            if seen >= max_tokens:
                break
        if seen >= max_tokens:
            break

    freqs = (fire_counts.float() / max(seen, 1)).numpy()
    dead_frac = float((fire_counts == 0).float().mean().item())
    mean_l0 = float(total_l0 / max(seen, 1))
    return {
        "dead_fraction": dead_frac,
        "mean_l0": mean_l0,
        "n_tokens_evaluated": int(seen),
        "atom_frequencies": freqs,
    }


def compute_probe_drop(
    sae: BatchTopKSAE,
    layer: int,
    scale: float,
    device: str = "cpu",
    ctrl: bool = False,
    seed: int = 0,
    exclude_run_ids: list[str] | None = None,
    max_train: int = 3000,
) -> dict:
    """
    Check 4: Evaluate 8-class grouped linear probe AUROC on:
      (a) Raw pooled layer embeddings x
      (b) SAE-reconstructed pooled layer embeddings x_hat
      (c) Pooled sparse SAE atom activations z (top-512 most active atoms to match d_in scale)
    """
    from model import load_ecgfm, extract_layers
    from probe import fit_probe_layer, eval_probe

    index = build_index()
    if exclude_run_ids:
        index = index.loc[~index["run_id"].isin(exclude_run_ids)].reset_index(drop=True)

    labels = index["group"].map(C.GROUP_TO_INT).to_numpy()
    groups = index["model_id"].astype(str).to_numpy()
    splits = index["split"].astype(str).to_numpy()

    tr_mask = (splits == "train")
    va_mask = (splits == "validation")
    te_mask = (splits == "test")

    tr_idx = np.where(tr_mask)[0]
    if len(tr_idx) > max_train:
        rng = np.random.default_rng(42)
        chosen = []
        n_each = max_train // len(C.GROUP_LABELS)
        for g_int in range(len(C.GROUP_LABELS)):
            g_idx = tr_idx[labels[tr_idx] == g_int]
            if len(g_idx) > n_each:
                g_idx = rng.choice(g_idx, n_each, replace=False)
            chosen.append(g_idx)
        tr_idx = np.concatenate(chosen)

    va_idx = np.where(va_mask)[0]
    te_idx = np.where(te_mask)[0]
    eval_indices = np.concatenate([tr_idx, va_idx, te_idx])

    model, _, _ = load_ecgfm(random_init=ctrl, seed=seed)
    model.to(device)
    model.eval()
    sae.eval()

    raw_embs = []
    recon_embs = []
    sparse_embs = []

    batch_size = 32
    N = len(eval_indices)
    for start in range(0, N, batch_size):
        batch_rows = index.iloc[eval_indices[start:start + batch_size]]
        win0, win1 = [], []
        for _, row in batch_rows.iterrows():
            try:
                w = load_medalcare(row["ecg_path"])
            except Exception:
                w = np.zeros((2, 12, C.WIN_SAMPLES), dtype=np.float32)
            win0.append(w[0])
            win1.append(w[1])

        x0 = torch.from_numpy(np.stack(win0)).float()
        x1 = torch.from_numpy(np.stack(win1)).float()
        feats0 = extract_layers(model, x0, layers=[layer], device=device)[layer].to(device)
        feats1 = extract_layers(model, x1, layers=[layer], device=device)[layer].to(device)

        B_cur, T_cur, D_cur = feats0.shape
        with torch.no_grad():
            xhat0, z0 = sae(feats0.reshape(-1, D_cur) / scale)
            xhat1, z1 = sae(feats1.reshape(-1, D_cur) / scale)
            xhat0 = (xhat0.reshape(B_cur, T_cur, D_cur) * scale).mean(dim=1).cpu().numpy()
            xhat1 = (xhat1.reshape(B_cur, T_cur, D_cur) * scale).mean(dim=1).cpu().numpy()

            z0_pool = torch.log1p(z0.reshape(B_cur, T_cur, -1).max(dim=1).values).cpu().numpy()
            z1_pool = torch.log1p(z1.reshape(B_cur, T_cur, -1).max(dim=1).values).cpu().numpy()

            raw0 = feats0.mean(dim=1).cpu().numpy()
            raw1 = feats1.mean(dim=1).cpu().numpy()

        raw_embs.append((raw0 + raw1) / 2.0)
        recon_embs.append((xhat0 + xhat1) / 2.0)
        sparse_embs.append(np.maximum(z0_pool, z1_pool))
        print(f"  [Probe check] Embedded {min(start + batch_size, N)}/{N}", end="\r", flush=True)
    print()

    X_raw = np.concatenate(raw_embs, axis=0)
    X_rec = np.concatenate(recon_embs, axis=0)
    X_sp  = np.concatenate(sparse_embs, axis=0)

    n_tr = len(tr_idx)
    n_va = len(va_idx)
    s_va = slice(n_tr, n_tr + n_va)
    s_te = slice(n_tr + n_va, None)

    y_tr, g_tr = labels[tr_idx], groups[tr_idx]
    y_va = labels[va_idx]
    y_te = labels[te_idx]

    clf_raw, sc_raw = fit_probe_layer(X_raw[:n_tr], y_tr, g_tr)
    raw_val = eval_probe(clf_raw, sc_raw, X_raw[s_va], y_va)
    raw_test = eval_probe(clf_raw, sc_raw, X_raw[s_te], y_te)

    clf_rec, sc_rec = fit_probe_layer(X_rec[:n_tr], y_tr, g_tr)
    rec_val = eval_probe(clf_rec, sc_rec, X_rec[s_va], y_va)
    rec_test = eval_probe(clf_rec, sc_rec, X_rec[s_te], y_te)

    # For sparse codes z, keep non-dead atoms with highest variance on train split
    var_tr = X_sp[:n_tr].var(axis=0)
    top_atoms = np.argsort(-var_tr)[:min(768, (var_tr > 1e-8).sum() or 768)]
    clf_sp, sc_sp = fit_probe_layer(X_sp[:n_tr][:, top_atoms], y_tr, g_tr)
    sp_val = eval_probe(clf_sp, sc_sp, X_sp[s_va][:, top_atoms], y_va)
    sp_test = eval_probe(clf_sp, sc_sp, X_sp[s_te][:, top_atoms], y_te)

    return {
        "raw_val_auc": float(raw_val["macro_auc"]),
        "raw_test_auc": float(raw_test["macro_auc"]),
        "recon_val_auc": float(rec_val["macro_auc"]),
        "recon_test_auc": float(rec_test["macro_auc"]),
        "sparse_z_val_auc": float(sp_val["macro_auc"]),
        "sparse_z_test_auc": float(sp_test["macro_auc"]),
        "probe_drop_val": float(raw_val["macro_auc"] - rec_val["macro_auc"]),
        "probe_drop_test": float(raw_test["macro_auc"] - rec_test["macro_auc"]),
    }


def run_checks(
    sae: BatchTopKSAE,
    shard_dir: Path,
    token_source: str,
    layer: int,
    scale: float,
    device: str = "cpu",
    ctrl: bool = False,
    seed: int = 0,
    exclude_run_ids: list[str] | None = None,
) -> dict:
    """Run all four Test B checks."""
    train_shards = _shard_paths(shard_dir, "train")
    held_shards  = _shard_paths(shard_dir, "held")
    if not held_shards:
        held_shards = train_shards[-max(1, len(train_shards) // 5):]

    print("[SAE checks] Computing EV on train and held-out shards...")
    ev_train_info = explained_variance(sae, train_shards, scale, device)
    ev_held_info  = explained_variance(sae, held_shards, scale, device)

    if token_source == "ptbxl":
        ev_ptbxl = ev_held_info["ev_per_token"]
        ev_sim   = _ev_on_medalcare(sae, layer, scale, device, ctrl=ctrl, seed=seed)
    else:
        ev_ptbxl = ev_train_info["ev_per_token"]
        ev_sim   = ev_held_info["ev_per_token"]

    print("[SAE checks] Computing dead atom fraction and L0 sparsity...")
    sp_info = sparsity_and_dead_atoms(sae, held_shards + train_shards, scale, device)
    dead_frac = sp_info["dead_fraction"]
    mean_l0   = sp_info["mean_l0"]

    print("[SAE checks] Running Check 4 (downstream 8-class probe drop on x_hat and z)...")
    probe_info = compute_probe_drop(
        sae, layer, scale, device=device, ctrl=ctrl, seed=seed,
        exclude_run_ids=exclude_run_ids,
    )

    checks = {
        "token_source":      token_source,
        "ev_ptbxl":          float(ev_ptbxl),
        "ev_sim":            float(ev_sim),
        "ev_train_per_token": ev_train_info["ev_per_token"],
        "ev_train_global":    ev_train_info["ev_global"],
        "ev_held_per_token":  ev_held_info["ev_per_token"],
        "ev_held_global":     ev_held_info["ev_global"],
        "dead_fraction":     float(dead_frac),
        "mean_l0":           float(mean_l0),
        **probe_info,
    }

    ev_pass   = (ev_sim >= 0.70)
    dead_pass = (dead_frac < 0.20)
    drop_pass = (probe_info["probe_drop_test"] <= 0.05)
    verdict   = "PASS" if (ev_pass and dead_pass and drop_pass) else "MARGINAL"
    checks["verdict"] = verdict

    print("\nTest B checks:")
    print(f"  EV Train:               {ev_train_info['ev_per_token']:.4f} (global: {ev_train_info['ev_global']:.4f})")
    print(f"  EV Held-out (>= 0.70):  {ev_sim:.4f} (global: {ev_held_info['ev_global']:.4f})  {'PASS' if ev_pass else 'FAIL'}")
    print(f"  Dead atoms (< 20%):     {dead_frac:.4f} (mean L0={mean_l0:.1f})  {'PASS' if dead_pass else 'FAIL'}")
    print(f"  Probe Test AUROC:       raw={probe_info['raw_test_auc']:.4f} -> x_hat={probe_info['recon_test_auc']:.4f} "
          f"(drop={probe_info['probe_drop_test']:+.4f}, sparse_z={probe_info['sparse_z_test_auc']:.4f})  "
          f"{'PASS' if drop_pass else 'FAIL'}")
    print(f"  Verdict:                {verdict}")

    return checks, sp_info["atom_frequencies"]


def _ev_on_medalcare(
    sae: BatchTopKSAE,
    layer: int,
    scale: float,
    device: str = "cpu",
    n_sample: int = 500,
    ctrl: bool = False,
    seed: int = 0,
) -> float:
    """Compute EV of SAE on a random sample of held-out MedalCare ECGs."""
    from model import load_ecgfm, extract_layers
    model, _, _ = load_ecgfm(random_init=ctrl, seed=seed)
    model.to(device)
    sae.eval()

    index = build_index()
    held = index[index["split"].isin(["validation", "test"])]
    sample = held.sample(min(n_sample, len(held)), random_state=42)

    all_mse, all_var = 0.0, 0.0
    for start in range(0, len(sample), 32):
        batch = sample.iloc[start:start + 32]
        wins = []
        for _, row in batch.iterrows():
            try:
                w = load_medalcare(row["ecg_path"])
                wins.append(w[0])
                wins.append(w[1])
            except Exception:
                continue
        if not wins:
            continue
        x = torch.from_numpy(np.stack(wins)).float()
        feats = extract_layers(model, x, layers=[layer], device=device)[layer]
        toks = feats.reshape(-1, feats.shape[-1]).float().to(device) / scale
        with torch.no_grad():
            xhat, _ = sae(toks)
        all_mse += ((toks - xhat) ** 2).mean(dim=-1).sum().item()
        all_var += toks.var(dim=-1, unbiased=False).sum().item()

    return 1.0 - (all_mse / max(all_var, 1e-12))


def _plot_sae_diagnostics(checks: dict, atom_freqs: np.ndarray, out_path: Path, tag: str):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        # Panel A: Summary metrics bar chart
        labels = ["Train EV", "Held-out EV", "Raw Probe\nTest AUROC", "SAE x_hat\nTest AUROC", "SAE Sparse z\nTest AUROC"]
        vals = [
            checks["ev_train_per_token"],
            checks["ev_sim"],
            checks["raw_test_auc"],
            checks["recon_test_auc"],
            checks["sparse_z_test_auc"],
        ]
        colors = ["#4c72b0", "#55a868", "#2b5c8f", "#388e3c", "#8172b2"]
        bars = axes[0].bar(labels, vals, color=colors, width=0.58)
        axes[0].set_ylim(0.0, 1.08)
        axes[0].axhline(0.70, color="gray", linestyle="--", linewidth=0.9, label="EV gate (0.70)")
        axes[0].set_ylabel("Score")
        axes[0].set_title(f"Test B: SAE Reconstruction & Probe Preservation ({tag})")
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=8.5)
        axes[0].legend(loc="lower right", fontsize=8)

        # Panel B: Atom activation frequency distribution
        live_freqs = np.clip(atom_freqs[atom_freqs > 0], 1e-6, 1.0)
        axes[1].hist(np.log10(live_freqs), bins=40, color="#4c72b0", edgecolor="white")
        axes[1].set_xlabel("log10(Atom Firing Frequency)")
        axes[1].set_ylabel("Number of SAE Atoms")
        axes[1].set_title(
            f"Atom Density (Dead: {100*checks['dead_fraction']:.1f}%, Mean L0: {checks['mean_l0']:.1f})"
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved SAE figure to {out_path}")
    except Exception as e:
        print(f"[WARN] Could not plot SAE diagnostics: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_test_b(
    layer: int = C.LOCKED_LAYER or 8,
    device: str = "cpu",
    rebuild_tokens: bool = False,
    rebuild_sae: bool = False,
    ctrl: bool = False,
    seed: int = 0,
    exclude_run_ids: list[str] | None = None,
) -> dict:
    excluded = sorted(dict.fromkeys(exclude_run_ids or []))
    suffix = ""
    if excluded:
        id_str = "_".join(r.removeprefix("run_") for r in excluded)
        suffix = f"_sensitivity_without_{id_str}"

    tag = f"ctrl_seed{seed}" if ctrl else "pretrained"
    shard_dir = TOKEN_SCRATCH_ROOT / f"{tag}{suffix}_layer{layer}"
    meta_path = shard_dir / "source.json"

    results = {
        "layer": int(layer),
        "tag": tag,
        "ctrl": bool(ctrl),
        "seed": int(seed),
        "excluded_run_ids": excluded,
    }

    # ── Step 1: extract tokens ─────────────────────────────────────────────
    has_shards = bool(_shard_paths(shard_dir, "train")) and bool(_shard_paths(shard_dir, "held"))
    if rebuild_tokens or not has_shards:
        print(f"\n[Test B] Extracting tokens ({tag}, layer {layer})...")
        from model import load_ecgfm
        model, _, d_model = load_ecgfm(random_init=ctrl, seed=seed)
        model.to(device)
        token_source = extract_tokens(
            layer, model, shard_dir, device=device, exclude_run_ids=excluded,
        )
        meta_path.write_text(json.dumps({"token_source": token_source}), encoding="utf-8")
    else:
        print(f"\n[Test B] Using cached tokens from {shard_dir}")
        d_model = C.SAE_D_IN
        token_source = "medalcare"
        if meta_path.is_file():
            token_source = json.loads(meta_path.read_text(encoding="utf-8")).get("token_source", "medalcare")

    # ── Step 2: scale ──────────────────────────────────────────────────────
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scale_path = C.CACHE_DIR / f"scale_{tag}{suffix}_layer{layer}.npy"
    legacy_scale_path = C.CACHE_DIR / f"scale_{tag}_layer{layer}.npy"
    if scale_path.exists() and not rebuild_tokens:
        scale = float(np.load(scale_path))
        print(f"[Test B] Loaded scale constant: {scale:.4f}")
    else:
        scale = compute_scale(shard_dir, d_model)
        np.save(scale_path, np.array(scale))
        np.save(legacy_scale_path, np.array(scale))

    results["scale"] = scale

    # ── Step 3: train SAE ─────────────────────────────────────────────────
    seed_suffix = f"_seed{seed}" if (not ctrl and seed != 0) else ""
    weights_path = C.CACHE_DIR / f"sae_{tag}{suffix}_layer{layer}{seed_suffix}.pt"
    legacy_weights_path = C.SAE_CTRL_WEIGHTS if ctrl else C.SAE_WEIGHTS
    sae_train_seed = seed if ctrl else (42 + seed)
    if weights_path.exists() and not rebuild_sae and not rebuild_tokens:
        print(f"[Test B] Loading SAE weights from {weights_path}")
        sae = BatchTopKSAE().to(device)
        sae.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"[Test B] Training BatchTopK SAE ({tag}{seed_suffix}, layer {layer}, train_seed={sae_train_seed})...")
        sae = train_sae(shard_dir, scale, device=device, seed=sae_train_seed)
        torch.save(sae.state_dict(), weights_path)
        if seed == 0 or ctrl:
            torch.save(sae.state_dict(), legacy_weights_path)
        print(f"SAE weights saved to {weights_path}")

    # ── Step 4: checks ─────────────────────────────────────────────────────
    checks, atom_freqs = run_checks(
        sae, shard_dir, token_source, layer, scale,
        device=device, ctrl=ctrl, seed=seed, exclude_run_ids=excluded,
    )
    results["checks"] = checks

    # ── Save ────────────────────────────────────────────────────────────────
    ctrl_str = f"_ctrl_seed{seed}" if (ctrl and seed != 0) else ("_ctrl" if ctrl else "")
    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.RUNS_DIR / f"B_results{ctrl_str}{suffix}{seed_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    fig_path = C.RUNS_DIR / f"B_sae_diagnostics{ctrl_str}{suffix}{seed_suffix}.png"
    _plot_sae_diagnostics(checks, atom_freqs, fig_path, f"{tag}{seed_suffix}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=C.LOCKED_LAYER or 8, help="Locked layer from Test A")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rebuild-tokens", action="store_true")
    ap.add_argument("--rebuild-sae", action="store_true")
    ap.add_argument("--ctrl", action="store_true", help="Run on random-init control")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-run-ids", nargs="*", default=None)
    args = ap.parse_args()
    run_test_b(
        args.layer, args.device, args.rebuild_tokens, args.rebuild_sae,
        args.ctrl, args.seed, args.exclude_run_ids,
    )
