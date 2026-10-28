"""Build a compact, lossily quantized MedalCare bundle for transfer to a Pod."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import load_medalcare


FORMAT_VERSION = 1
SHARD_RECORDS = 128


def _validate_index(index: pd.DataFrame) -> None:
    required = {"ecg_path", "group", "split", "model_id", "source_shape_id", "run_id"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Dataset index is missing required columns: {sorted(missing)}")
    if index.empty:
        raise ValueError("Dataset index has no records.")
    if index[list(required)].isna().any().any():
        raise ValueError("Required index metadata contains null values.")
    if set(index["group"].unique()) != set(C.GROUP_LABELS):
        raise ValueError(
            f"Index group labels differ from expected: {sorted(index['group'].unique())}"
        )
    expected_splits = {"train", "validation", "test"}
    if set(index["split"].unique()) != expected_splits:
        raise ValueError(f"Index is missing an expected split: {sorted(index['split'].unique())}")
    split_counts = index.groupby("model_id")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError(
            f"{int((split_counts > 1).sum())} model IDs appear across predefined splits."
        )
    for split in expected_splits:
        missing_groups = set(C.GROUP_LABELS) - set(index.loc[index["split"] == split, "group"])
        if missing_groups:
            raise ValueError(f"Split {split} is missing groups: {sorted(missing_groups)}")

    missing_files = [path for path in index["ecg_path"] if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} indexed ECG files are unavailable; examples: "
            + ", ".join(map(str, missing_files[:5]))
        )


def select_pilot(
    index: pd.DataFrame,
    train_models: int = 5,
    heldout_models: int = 1,
    per_model: int = 10,
    seed: int = 0,
    exclude_run_ids: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict]:
    """Deterministic split-preserving subset: per group, N models per split, capped ECGs per model.

    Never moves a model between splits. Prefers models that have at least `per_model` ECGs.
    """
    unknown = set(exclude_run_ids) - set(index["run_id"])
    if unknown:
        raise ValueError(f"Unknown run IDs requested for exclusion: {sorted(unknown)}")
    pool = index.loc[~index["run_id"].isin(exclude_run_ids)]
    rng = np.random.default_rng(seed)
    wanted = {"train": train_models, "validation": heldout_models, "test": heldout_models}
    picked: list[pd.DataFrame] = []
    selected_models: dict[str, dict[str, list[str]]] = {}
    for group in sorted(pool["group"].unique()):
        for split, n_models in wanted.items():
            rows = pool[(pool["group"] == group) & (pool["split"] == split)]
            sizes = rows.groupby("model_id").size().sort_index()
            if len(sizes) < n_models:
                raise ValueError(
                    f"{group}/{split} has {len(sizes)} models after exclusion; need {n_models}."
                )
            order = rng.permutation(len(sizes))
            candidates = sizes.iloc[order]
            # stable sort: models with enough ECGs first, shuffled order kept within each class
            candidates = candidates.iloc[np.argsort(candidates.to_numpy() < per_model, kind="stable")]
            chosen = sorted(candidates.index[:n_models])
            selected_models.setdefault(group, {})[split] = chosen
            for model_id in chosen:
                model_rows = rows[rows["model_id"] == model_id].sort_values("ecg_path")
                take = min(per_model, len(model_rows))
                picked.append(model_rows.iloc[np.sort(rng.choice(len(model_rows), take, replace=False))])
    subset = pd.concat(picked)
    info = {
        "purpose": "pilot-only; exploratory; not a Test A result",
        "seed": seed,
        "train_models_per_group": train_models,
        "heldout_models_per_group": heldout_models,
        "max_ecgs_per_model": per_model,
        "excluded_run_ids": sorted(exclude_run_ids),
        "records": int(len(subset)),
        "counts_by_group_split": {
            g: subset[subset["group"] == g]["split"].value_counts().to_dict()
            for g in sorted(subset["group"].unique())
        },
        "selected_model_ids": selected_models,
    }
    return subset, info


def _make_shard(records: list[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, normalized in records:
        scales = np.max(np.abs(normalized), axis=1) / 127.0
        scales = np.maximum(scales, np.finfo(np.float32).tiny).astype(np.float32)
        quantized = np.clip(
            np.rint(normalized / scales[:, None]), -127, 127
        ).astype(np.int8)
        arrays[key] = quantized
        arrays[f"{key}_scale"] = scales
    return arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_medalcare(
    output: Path,
    index_path: Path = C.INDEX_PARQUET,
    shard_records: int = SHARD_RECORDS,
    pilot: dict | None = None,
) -> dict:
    if shard_records < 1:
        raise ValueError("shard_records must be positive.")
    index = pd.read_parquet(index_path)
    _validate_index(index)
    pilot_info = None
    if pilot is not None:
        index, pilot_info = select_pilot(index, **pilot)
        _validate_index(index)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing bundle: {output}. "
            "Choose a new output path or move the existing file first."
        )

    manifest = index.copy()
    manifest["ecg_path"] = ""
    expected_records = len(index)
    normalized_values = 0
    squared_error = 0.0
    max_error = 0.0
    max_abs_value = 0.0
    temp_archive = output.with_name(f".{output.name}.{os.getpid()}.tmp")

    try:
        with tempfile.TemporaryDirectory(prefix="ecg-gate-package-") as temp_dir:
            temp_root = Path(temp_dir)
            metadata = {
                "format_version": FORMAT_VERSION,
                "record_count": expected_records,
                "groups": C.GROUP_LABELS,
                "expected_total_ecgs": C.EXPECTED_TOTAL_ECGS,
                "shard_records": shard_records,
                "waveform": {
                    "shape": [12, 5000],
                    "source_preprocessing": "per-lead z-score using std + 1e-8",
                    "storage_dtype": "int8",
                    "scale_dtype": "float32 per ECG lead",
                    "decode": "normalized_float32 = int8_values * per_lead_scale",
                    "quantization": "round(normalized / (max_abs_per_lead / 127))",
                },
            }

            if pilot_info is not None:
                metadata["pilot"] = pilot_info

            with zipfile.ZipFile(
                temp_archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as bundle:
                records: list[tuple[str, np.ndarray]] = []
                shard_number = 0
                for position, (_, row) in enumerate(index.iterrows()):
                    source_path = Path(row["ecg_path"])
                    windows = load_medalcare(source_path)
                    normalized = np.concatenate((windows[0], windows[1]), axis=1)
                    if normalized.shape != (12, 5000) or not np.isfinite(normalized).all():
                        raise ValueError(f"Invalid preprocessed ECG at row {position}: {source_path}")

                    scales = np.maximum(
                        np.max(np.abs(normalized), axis=1) / 127.0,
                        np.finfo(np.float32).tiny,
                    )
                    quantized = np.clip(
                        np.rint(normalized / scales[:, None]), -127, 127
                    ).astype(np.int8)
                    restored = quantized.astype(np.float32) * scales.astype(np.float32)[:, None]
                    error = normalized - restored
                    squared_error += float(np.square(error, dtype=np.float64).sum())
                    normalized_values += normalized.size
                    max_error = max(max_error, float(np.max(np.abs(error))))
                    max_abs_value = max(max_abs_value, float(np.max(np.abs(normalized))))

                    shard_number, slot = divmod(position, shard_records)
                    key = f"ecg_{slot:04d}"
                    relative_shard = f"shards/shard_{shard_number:05d}.npz"
                    manifest.at[index.index[position], "ecg_path"] = f"{relative_shard}::{key}"
                    records.append((key, normalized))

                    is_last = position + 1 == expected_records
                    if len(records) == shard_records or is_last:
                        shard_path = temp_root / f"shard_{shard_number:05d}.npz"
                        np.savez_compressed(shard_path, **_make_shard(records))
                        bundle.write(shard_path, arcname=relative_shard)
                        records.clear()
                        print(
                            f"Packaged {min(position + 1, expected_records)}/"
                            f"{expected_records} ECGs",
                            flush=True,
                        )

                manifest_path = temp_root / "manifest.parquet"
                manifest.to_parquet(manifest_path, index=False)
                bundle.write(manifest_path, arcname="manifest.parquet")

                metadata["quantization_stats"] = {
                    "rmse": math.sqrt(squared_error / normalized_values),
                    "max_abs_error": max_error,
                    "max_abs_normalized_value": max_abs_value,
                }
                metadata_path = temp_root / "bundle.json"
                metadata_path.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                bundle.write(metadata_path, arcname="bundle.json")

        temp_archive.replace(output)
    except Exception:
        temp_archive.unlink(missing_ok=True)
        raise

    result = {
        "output": str(output),
        "bytes": output.stat().st_size,
        "gib": output.stat().st_size / 1024**3,
        "records": expected_records,
        "sha256": _sha256(output),
        "rmse": math.sqrt(squared_error / normalized_values),
        "max_abs_error": max_error,
        "max_abs_normalized_value": max_abs_value,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output ZIP path.")
    parser.add_argument("--index", type=Path, default=C.INDEX_PARQUET)
    parser.add_argument("--shard-records", type=int, default=SHARD_RECORDS)
    parser.add_argument("--pilot", action="store_true",
                        help="Export a small deterministic split-preserving pilot subset.")
    parser.add_argument("--pilot-train-models", type=int, default=5)
    parser.add_argument("--pilot-heldout-models", type=int, default=1)
    parser.add_argument("--pilot-per-model", type=int, default=10)
    parser.add_argument("--pilot-seed", type=int, default=0)
    parser.add_argument("--exclude-run-ids", nargs="*", default=["run_S64", "run_S67"],
                        help="Pilot only: run IDs to drop (default: the two split-crossing runs).")
    args = parser.parse_args()
    pilot = None
    if args.pilot:
        pilot = {
            "train_models": args.pilot_train_models,
            "heldout_models": args.pilot_heldout_models,
            "per_model": args.pilot_per_model,
            "seed": args.pilot_seed,
            "exclude_run_ids": tuple(args.exclude_run_ids),
        }
    result = package_medalcare(args.output, args.index, args.shard_records, pilot)
    print("\nBundle complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
