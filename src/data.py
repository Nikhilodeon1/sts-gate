"""
data.py – preprocessing and dataset index building.

Covers:
  - load_medalcare(csv_path) -> (2, 12, 2500) float32
  - load_ptbxl(wfdb_path_no_ext) -> (2, 12, 2500) float32
  - build_index() -> index.parquet
  - verify_index() -> mapping_report.md
"""
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src/ to path so config is importable when run directly
sys.path.insert(0, str(Path(__file__).parent))
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers (shared across all datasets)
# ─────────────────────────────────────────────────────────────────────────────

def reorder_by_name(sig: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Reorder wfdb-loaded signal by lead name.
    sig: (T, 12) time-first from wfdb.rdsamp
    returns: (12, T) in LEADS order
    """
    up = [n.upper().replace(" ", "") for n in names]
    lead_up = [l.upper().replace(" ", "") for l in C.LEADS]
    idx = [up.index(l) for l in lead_up]
    return sig[:, idx].T   # -> (12, T)


def zscore(x: np.ndarray) -> np.ndarray:
    """
    Per-lead z-score.  x: (12, T)
    Matches the ECG-FM Standardize() transform which operates per lead.
    """
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1,  keepdims=True)
    return (x - mu) / (std + 1e-8)


def make_windows(x: np.ndarray) -> np.ndarray:
    """
    (12, 5000) -> (2, 12, 2500): two non-overlapping 5-second windows.
    """
    assert x.shape == (12, 5000), f"Expected (12,5000), got {x.shape}"
    return np.stack([x[:, :2500], x[:, 2500:5000]]).astype(np.float32)


def load_medalcare(csv_path: str | Path) -> np.ndarray:
    """
    Load one source CSV or a compact ECG bundle record.
    CSV has no header; rows are leads in LEADS order; 5000 columns (samples).
    Returns (2, 12, 2500) float32.
    """
    path_text = str(csv_path)
    if "::" in path_text:
        shard_path, _, array_key = path_text.partition("::")
        if not array_key:
            raise ValueError(f"Missing array key in bundled ECG path: {path_text}")
        with np.load(shard_path, allow_pickle=False) as shard:
            quantized = shard[array_key]
            scale = shard[f"{array_key}_scale"]
        if quantized.shape != (12, 5000) or scale.shape != (12,):
            raise ValueError(
                f"Unexpected bundled ECG shape/scales in {path_text}: "
                f"{quantized.shape} / {scale.shape}"
            )
        normalized = quantized.astype(np.float32) * scale.astype(np.float32)[:, None]
        return make_windows(normalized)

    raw = np.loadtxt(str(csv_path), delimiter=",")   # (12, 5000)
    if raw.shape != (12, 5000):
        raise ValueError(f"Unexpected MedalCare shape {raw.shape} in {csv_path}")
    return make_windows(zscore(raw))


def load_ptbxl(wfdb_path: str | Path) -> np.ndarray:
    """
    Load one PTB-XL record (no extension, e.g. .../00001_hr).
    Returns (2, 12, 2500) float32.
    """
    import wfdb
    sig, meta = wfdb.rdsamp(str(wfdb_path))   # sig: (T, 12)
    x = reorder_by_name(sig, meta["sig_name"])  # (12, T)
    # records500 are already at 500 Hz (5000 samples); crop/pad to be safe
    if x.shape[1] > 5000:
        x = x[:, :5000]
    elif x.shape[1] < 5000:
        pad = np.zeros((12, 5000 - x.shape[1]), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=1)
    return make_windows(zscore(x))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter file parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_param_file(path: Path) -> dict:
    """Parse MedalCare key/value files, including values with units."""
    d = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        key = key.strip().replace('"', "").replace("'", "")
        value = value.strip().replace('"', "").replace("'", "")
        if value.casefold() in {"", "na", "nan", "null"}:
            d[key] = np.nan
            continue

        numeric = re.fullmatch(
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"(?:\s*[A-Za-z%/]*)?",
            value,
        )
        d[key] = float(numeric.group(1)) if numeric else value
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Index builder (Experiment 1)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_medalcare_ecgs():
    """
    Yield dicts describing each MedalCare ECG file (filtered variant).
    siginfo.csv has two columns: info1 (atrial mat filename), info2 (run path).
    Some folders preserve info2's run number in filenames; others export
    sequentially numbered ECG and parameter files corresponding to siginfo row
    order. Model IDs combine the group label and the source geometry token.
    """
    noise_root  = C.MEDALCARE_NOISE
    param_root  = C.MEDALCARE_PARAMS

    if not noise_root.is_dir():
        raise FileNotFoundError(f"MedalCare noise directory not found: {noise_root}")

    # Top-level folders: sinus, lbbb, rbbb, avblock, lae, fam, iab, mi
    top_folders = [p for p in noise_root.iterdir() if p.is_dir()
                   and not p.name.startswith(".") and p.name != "__MACOSX"]

    for top in sorted(top_folders):
        # Determine sub-folders (mi has sub-sub-folders LAD_0.3 etc.)
        sub_candidates = list(top.iterdir())
        has_splits = any(d.name in ("train","validation","test")
                         for d in sub_candidates if d.is_dir())
        if has_splits:
            # Simple folder: sinus/train/run_S65/...
            walk_roots = [(top, top.name, top.name)]
        else:
            # MI folder: mi/LAD_0.3/train/run_S../...
            walk_roots = [
                (sub, f"{top.name}/{sub.name}", f"{top.name}/{sub.name}")
                for sub in sorted(sub_candidates)
                if sub.is_dir() and sub.name != "examples"
            ]

        for noise_dir, folder_key, mi_key in walk_roots:
            group = C.FOLDER_TO_GROUP.get(folder_key, "unknown")
            if group == "unknown":
                raise ValueError(f"Unrecognized MedalCare folder: {folder_key}")
            for split in ("train", "validation", "test"):
                split_dir = noise_dir / split
                if not split_dir.exists():
                    continue
                for run_dir in sorted(split_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    run_id = run_dir.name  # e.g. run_S65

                    siginfo_path = run_dir / "siginfo.csv"
                    if not siginfo_path.exists():
                        continue

                    # Corresponding parameter folder
                    # noise path:  MEDALCARE_NOISE/<folder>/<split>/<run_id>/
                    # param path:  MEDALCARE_PARAMS/<folder>/<split>/<run_id>/
                    rel = run_dir.relative_to(noise_root)
                    param_dir = param_root / rel

                    # Read siginfo
                    try:
                        si = pd.read_csv(siginfo_path, header=0)
                    except Exception as exc:
                        raise RuntimeError(f"Could not read siginfo file {siginfo_path}") from exc
                    required_columns = {"info1", "info2"}
                    missing_columns = required_columns - set(si.columns)
                    if missing_columns:
                        raise ValueError(
                            f"{siginfo_path} is missing required columns "
                            f"{sorted(missing_columns)}; found {list(si.columns)}"
                        )

                    # info2 paths are direct file IDs in some folders. In other
                    # folders the converted files use sequential output names.
                    skipped_rows = []
                    for row_index, (_, row) in enumerate(si.iterrows(), start=1):
                        info1 = row.get("info1")
                        info2 = row.get("info2")
                        if pd.isna(info1) or pd.isna(info2):
                            skipped_rows.append(f"{row_index} (null siginfo)")
                            continue
                        atrial_mat = str(info1).strip()   # ecgs_..._shapeID_29_rot11_torsoID12.mat
                        run_path   = str(info2).strip()   # folder/split/run_id/run_NNNNNN
                        if not atrial_mat or not run_path:
                            skipped_rows.append(f"{row_index} (empty siginfo)")
                            continue

                        # Extract run number from run_path
                        run_num_match = re.search(r"run_(\d+)$", run_path)
                        if not run_num_match:
                            raise ValueError(
                                f"Unexpected run path in {siginfo_path}:{row_index}: {run_path!r}"
                            )
                        run_num   = run_num_match.group(1)

                        # Prefer info2's exact run ID. Converted folders may use
                        # sequential names aligned with siginfo row order.
                        direct_stem = f"run_{run_num}"
                        direct_ecg = run_dir / f"{direct_stem}_filtered.csv"
                        ordinal_stem = f"run_{row_index:06d}"
                        ordinal_ecg = run_dir / f"{row_index:06d}_filtered.csv"
                        if direct_ecg.exists():
                            run_stem = direct_stem
                            ecg_file = direct_ecg
                        else:
                            run_stem = ordinal_stem
                            ecg_file = ordinal_ecg
                        if not ecg_file.exists():
                            skipped_rows.append(f"{row_index} (filtered ECG not found)")
                            continue

                        # Keep the complete source token; not all shape IDs are numeric.
                        source_shape_match = re.search(r"shapeID_([^_]+)", atrial_mat)
                        source_shape_id = source_shape_match.group(1) if source_shape_match else None
                        if source_shape_id is None:
                            pwave_match = re.search(r"pwave_(\d+)", atrial_mat, re.IGNORECASE)
                            if pwave_match:
                                source_shape_id = f"pwave_{int(pwave_match.group(1)):03d}"

                        # Preserve the old numeric field for existing caches/consumers.
                        shape_match = re.search(r"shapeID_(\d+)", atrial_mat)
                        vent_geom   = int(shape_match.group(1)) if shape_match else -1

                        # Torso ID
                        torso_match = re.search(r"torsoID(\d+)", atrial_mat)
                        torso_id    = int(torso_match.group(1)) if torso_match else -1

                        # Parameter files
                        atrial_param_file = param_dir / f"{run_stem}_AtrialParameters.txt"
                        vent_param_file   = param_dir / f"{run_stem}_VentricularParameters.txt"

                        atrial_params = {}
                        vent_params   = {}
                        if atrial_param_file.exists():
                            atrial_params = _parse_param_file(atrial_param_file)
                        if vent_param_file.exists():
                            vent_params = _parse_param_file(vent_param_file)

                        model_id = (
                            f"{group}::{source_shape_id}"
                            if source_shape_id is not None else None
                        )
                        if model_id is None:
                            print(
                                f"  [WARN] Missing source geometry ID for "
                                f"{ecg_file}; model-based checks will fail."
                            )

                        record = {
                            "ecg_path":      str(ecg_file),
                            "folder":        folder_key,
                            "group":         group,
                            "split":         split,
                            "run_id":        run_id,
                            "run_num":       run_num,
                            "atrial_mat":    atrial_mat,
                            "model_id":      model_id,
                            "source_shape_id": source_shape_id,
                            "vent_geom":     vent_geom,
                            "torso_id":      torso_id,
                            "atrial_run":    run_id,
                            "vent_run":      run_id,
                        }
                        # Flatten atrial params with prefix "a_"
                        for k, v in atrial_params.items():
                            record[f"a_{k}"] = v
                        # Flatten ventricular params with prefix "v_"
                        for k, v in vent_params.items():
                            record[f"v_{k}"] = v

                        yield record

                    if skipped_rows:
                        print(
                            f"  [WARN] Skipped {len(skipped_rows)} rows in {siginfo_path}: "
                            + ", ".join(skipped_rows)
                        )


def build_index(force: bool = False) -> pd.DataFrame:
    """
    Build index.parquet: one row per MedalCare filtered ECG with all parameters.
    Cached; pass force=True to rebuild.
    """
    if C.ECG_BUNDLE_ROOT is not None:
        bundle_root = C.ECG_BUNDLE_ROOT.resolve()
        metadata_path = bundle_root / "bundle.json"
        manifest_path = bundle_root / "manifest.parquet"
        if not metadata_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"ECG bundle must contain bundle.json and manifest.parquet: {bundle_root}"
            )
        import json
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError(
                f"Unsupported ECG bundle format version: {metadata.get('format_version')!r}"
            )
        df = pd.read_parquet(manifest_path)
        if len(df) != metadata.get("record_count"):
            raise ValueError(
                f"Bundle manifest has {len(df)} rows; metadata declares "
                f"{metadata.get('record_count')}"
            )
        if "ecg_path" not in df or df["model_id"].isna().any():
            raise ValueError("Bundle manifest is missing ECG paths or model IDs.")

        resolved_paths = []
        missing_shards = set()
        for ref in df["ecg_path"].astype(str):
            relative_path, separator, array_key = ref.partition("::")
            if not separator or not array_key:
                raise ValueError(f"Invalid bundled ECG reference: {ref!r}")
            shard_path = (bundle_root / relative_path).resolve()
            if bundle_root not in shard_path.parents:
                raise ValueError(f"ECG bundle reference escapes bundle root: {ref!r}")
            if not shard_path.is_file():
                missing_shards.add(str(shard_path))
            resolved_paths.append(f"{shard_path}::{array_key}")
        if missing_shards:
            raise FileNotFoundError(
                f"ECG bundle is missing {len(missing_shards)} shard file(s): "
                + ", ".join(sorted(missing_shards)[:5])
            )
        df["ecg_path"] = resolved_paths
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(C.INDEX_PARQUET, index=False)
        print(f"Loaded {len(df)} ECG records from compact bundle {bundle_root}")
        return df

    if C.INDEX_PARQUET.exists() and not force:
        cached = pd.read_parquet(C.INDEX_PARQUET)
        required_columns = {"model_id", "source_shape_id"}
        if required_columns.issubset(cached.columns):
            valid_shape_ids = cached["source_shape_id"].notna()
            expected_model_ids = (
                cached["group"].astype(str) + "::" + cached["source_shape_id"].astype(str)
            )
            uses_composite_ids = (
                cached.loc[valid_shape_ids, "model_id"].astype(str).reset_index(drop=True)
                .equals(expected_model_ids.loc[valid_shape_ids].reset_index(drop=True))
                and cached.loc[~valid_shape_ids, "model_id"].isna().all()
            )
        else:
            uses_composite_ids = False
        if uses_composite_ids:
            print(f"Loading cached index from {C.INDEX_PARQUET}")
            return cached
        print("Cached index uses an outdated model-ID schema; rebuilding with group-scoped IDs.")

    print("Building dataset index (this may take a few minutes)...")
    rows = list(_iter_medalcare_ecgs())
    df = pd.DataFrame(rows)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(C.INDEX_PARQUET, index=False)
        print(f"Saved {len(df)} rows to {C.INDEX_PARQUET}")
    else:
        print(f"No MedalCare records found; leaving {C.INDEX_PARQUET} unset.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PTB-XL record list (Experiment 1 / Test B)
# ─────────────────────────────────────────────────────────────────────────────

def get_ptbxl_records(max_folder: int = 4000) -> pd.DataFrame:
    """
    Return records500 folders 00000-04000 with strat_fold from ptbxl_database.csv.
    Only includes files that are actually present on disk so partial downloads do not crash the pipeline.
    """
    if not C.PTBXL_DB_CSV.exists():
        return pd.DataFrame(columns=["ecg_id", "wfdb_path", "strat_fold", "scp_codes"])

    db = pd.read_csv(C.PTBXL_DB_CSV)
    db = db[db["filename_hr"].notna()].copy()
    db["wfdb_path"] = db["filename_hr"].apply(
        lambda p: str(C.PTBXL_ROOT / p)
    )
    db["wfdb_exists"] = db["wfdb_path"].map(lambda p: Path(p).exists())

    # Filter to folders 00000-04000 and to records actually available on disk.
    folder_int = db["filename_hr"].str.extract(r"records500/(\d+)/").astype(float)[0]
    db = db[(folder_int <= max_folder) & db["wfdb_exists"]].copy()
    return db[["ecg_id", "wfdb_path", "strat_fold", "scp_codes"]].reset_index(drop=True)


if __name__ == "__main__":
    # Quick smoke test
    import json
    df = build_index()
    print(f"\nTotal ECGs: {len(df)}")
    print("Columns:", list(df.columns[:20]), "...")
    print("\nGroup counts:")
    print(df["group"].value_counts())
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\nAnatomical model IDs per split:")
    print(df.groupby("split")["model_id"].nunique())

    # Check parameter coverage
    key_params = ["v_APD.min","v_APD.max","a_fibperc","a_cv_t.BulkTissue"]
    for p in key_params:
        if p in df.columns:
            n_valid = df[p].notna().sum()
            pct = 100*n_valid/len(df)
            print(f"  {p}: {n_valid}/{len(df)} ({pct:.1f}%)")
