"""
exp1_index.py – Experiment 1: build index.parquet and write mapping_report.md.

Checks:
  1. Count ECGs per subfolder and variant.
  2. Coverage: fraction of ECGs with complete parameter rows.
  3. Range checks against paper values.
  4. Geometry disjointness across splits.
  5. Plot 3 random ECGs per group (filtered and raw).

Run:
    python src/exp1_index.py [--force-rebuild]
"""
import sys, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data import build_index, load_medalcare


# Expected parameter ranges from paper Tables 1-4
RANGE_CHECKS = {
    "v_APD.min":  (140, 185),     # paper: 150-175 ms (allow small margin)
    "v_APD.max":  (215, 260),     # paper: 225-250 ms
    "a_fibperc":  (0,   50),      # FAM: 5-45%
}


def run_exp1(force_rebuild: bool = False):

    # ── Build index ────────────────────────────────────────────────────────
    df = build_index(force=force_rebuild)
    total = len(df)
    print(f"\nTotal MedalCare ECGs indexed: {total} / {C.EXPECTED_TOTAL_ECGS}")

    # ── Expected-count coverage ────────────────────────────────────────────
    expected_by_group = {}
    for folder, expected in C.EXPECTED_FOLDER_COUNTS.items():
        group = C.FOLDER_TO_GROUP[folder]
        expected_by_group[group] = expected_by_group.get(group, 0) + expected

    indexed_by_group = df.groupby("group").size()
    group_coverage = pd.DataFrame([
        {
            "group": group,
            "indexed": int(indexed_by_group.get(group, 0)),
            "expected": expected,
            "pct": 100 * indexed_by_group.get(group, 0) / expected,
        }
        for group, expected in sorted(expected_by_group.items())
    ])

    key_params = ["v_APD.min", "v_APD.max", "a_cv_t.BulkTissue", "a_fibperc"]
    coverage_rows = []
    for param in key_params:
        n_valid = int(df[param].notna().sum()) if param in df.columns else 0
        coverage_rows.append({
            "param": param,
            "valid": n_valid,
            "expected": C.EXPECTED_TOTAL_ECGS,
            "pct": 100 * n_valid / C.EXPECTED_TOTAL_ECGS,
        })
    cov_df = pd.DataFrame(coverage_rows)
    overall_cov = min(group_coverage["pct"].min(), cov_df["pct"].min())
    cov_verdict = "PASS" if overall_cov >= 90 else "FAIL"
    print(f"\nMinimum expected group/parameter coverage: {overall_cov:.1f}%  [{cov_verdict}]")
    print("\nIndexed versus expected ECGs per group:")
    print(group_coverage.to_string(index=False, formatters={"pct": "{:.1f}%".format}))

    # ── Group / split counts ───────────────────────────────────────────────
    group_split = (
        df.groupby(["group", "split"]).size().unstack(fill_value=0)
        .reindex(index=C.GROUP_LABELS, columns=["train", "validation", "test"], fill_value=0)
    )
    print("\nECGs per group and split:")
    print(group_split.to_string())

    # ── Range checks ──────────────────────────────────────────────────────
    range_issues = []
    for param, (lo, hi) in RANGE_CHECKS.items():
        if param not in df.columns:
            continue
        vals = df[param].dropna()
        out_of_range = ((vals < lo) | (vals > hi)).sum()
        if out_of_range > 0:
            range_issues.append(f"  {param}: {out_of_range} values outside [{lo}, {hi}]")
    if range_issues:
        print("\nRange check issues (possible join bugs):")
        for issue in range_issues:
            print(issue)
    else:
        print("\nAll range checks passed.")

    # ── Anatomical model disjointness ──────────────────────────────────────
    model_ids = df["model_id"] if "model_id" in df else pd.Series(dtype="object")
    missing_model_ids = int(model_ids.isna().sum())
    valid_models = df.dropna(subset=["model_id"])
    model_split_counts = valid_models.groupby("model_id")["split"].nunique()
    model_group_counts = valid_models.groupby("model_id")["group"].nunique()
    overlap = model_split_counts[model_split_counts > 1]
    geom_verdict = "PASS" if missing_model_ids == 0 and overlap.empty else "FAIL"
    print(
        f"\nAnatomical model IDs: {model_ids.nunique()} unique; "
        f"{missing_model_ids} missing; {len(overlap)} occur in multiple splits."
    )
    print("Groups represented per model ID (group count -> number of IDs):")
    print(model_group_counts.value_counts().sort_index().to_dict())
    if not overlap.empty:
        print("Model IDs appearing in multiple splits:")
        print(overlap)

    # ── Plot sample ECGs ───────────────────────────────────────────────────
    _plot_sample_ecgs(df)

    # ── Write mapping_report.md ───────────────────────────────────────────
    _write_report(
        df, total, overall_cov, cov_verdict, cov_df, group_coverage,
        group_split, range_issues, geom_verdict, overlap,
        model_group_counts, missing_model_ids,
    )

    verdict = "PASS" if overall_cov >= 90 and geom_verdict == "PASS" else "FAIL"
    print(f"\nExp 1 verdict: {verdict}")
    return verdict


def _plot_sample_ecgs(df: pd.DataFrame, n_per_group: int = 3):
    """Plot sample ECGs for visual sanity check."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = C.RUNS_DIR / "exp1_ecg_plots"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for group in C.GROUP_LABELS:
            group_df = df[df["group"] == group]
            if group_df.empty:
                print(f"  [WARN] No ECGs indexed for group '{group}'; skipping plot.")
                continue
            # Sample from filtered files
            sample = group_df.sample(min(n_per_group, len(group_df)), random_state=42)

            fig, axes = plt.subplots(len(sample), 12, figsize=(24, 2*len(sample)))
            if len(sample) == 1:
                axes = axes[np.newaxis, :]

            for row_i, (_, row) in enumerate(sample.iterrows()):
                try:
                    arr = np.loadtxt(row["ecg_path"], delimiter=",")  # (12, 5000)
                    for lead_i in range(12):
                        axes[row_i, lead_i].plot(arr[lead_i, :2500], linewidth=0.5)
                        axes[row_i, lead_i].set_xticks([]); axes[row_i, lead_i].set_yticks([])
                        if row_i == 0:
                            axes[row_i, lead_i].set_title(C.LEADS[lead_i], fontsize=7)
                except Exception as e:
                    print(f"  [WARN] Could not plot {row['ecg_path']}: {e}")

            fig.suptitle(f"Group: {group} (filtered, window 1)", fontsize=10)
            out_path = fig_dir / f"{group}.png"
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        print(f"ECG plots saved to {fig_dir}")
    except Exception as e:
        print(f"[WARN] Could not plot ECGs: {e}")


def _write_report(df, total, overall_cov, cov_verdict, cov_df, group_coverage,
                  group_split, range_issues, geom_verdict, overlap,
                  model_group_counts, missing_model_ids):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_path = C.PROJECT_ROOT / "mapping_report.md"
    missing_groups = sorted(set(C.GROUP_LABELS) - set(df["group"].dropna()))
    indexed_by_folder = df.groupby("folder").size()
    folder_coverage = pd.DataFrame([
        {
            "folder": folder,
            "indexed": int(indexed_by_folder.get(folder, 0)),
            "expected": expected,
            "missing": max(0, expected - int(indexed_by_folder.get(folder, 0))),
        }
        for folder, expected in C.EXPECTED_FOLDER_COUNTS.items()
    ])
    underfilled_folders = folder_coverage[folder_coverage["missing"] > 0]
    group_counts = model_group_counts.value_counts().sort_index().to_dict()
    final_verdict = "PASS" if overall_cov >= 90 and geom_verdict == "PASS" else "FAIL"
    run_split_counts = df.groupby("run_id")["split"].nunique()
    runs_across_splits = run_split_counts[run_split_counts > 1]
    source_shape_groups = (
        df.dropna(subset=["source_shape_id"])
        .groupby("source_shape_id")["group"].nunique()
        if "source_shape_id" in df else pd.Series(dtype="int64")
    )
    model_details = (
        df.dropna(subset=["model_id"])
        .groupby("model_id")
        .agg(
            groups=("group", lambda values: ", ".join(sorted(set(values)))),
            splits=("split", lambda values: ", ".join(sorted(set(values)))),
        )
    )
    split_overlap_details = model_details.reindex(overlap.index)

    lines = [
        f"# MedalCare-XL Mapping Report",
        f"Generated: {ts}",
        "",
        f"## Summary",
        f"- Total ECGs indexed: **{total} / {C.EXPECTED_TOTAL_ECGS}** ({100 * total / C.EXPECTED_TOTAL_ECGS:.1f}%)",
        f"- Minimum expected group/parameter coverage: **{overall_cov:.1f}%** [{cov_verdict}]",
        f"- Anatomical model split disjointness: **{geom_verdict}**",
        f"- **Exp 1 verdict: {final_verdict}**",
        f"- Groups with no indexed records: **{', '.join(missing_groups) if missing_groups else 'None'}**",
        f"- ECG rows without a model ID: **{missing_model_ids}**",
        "",
        "## Indexed versus expected ECGs per group",
        group_coverage.to_markdown(index=False, floatfmt=".1f"),
        "",
        "## Indexed versus expected ECGs per folder",
        folder_coverage.to_markdown(index=False),
        "",
        "## ECGs per group and split",
        group_split.to_markdown() if hasattr(group_split, "to_markdown") else str(group_split),
        "",
        "## Parameter coverage",
        cov_df.to_markdown(index=False) if (not cov_df.empty and hasattr(cov_df, "to_markdown"))
              else str(cov_df),
        "",
        "## Range check issues",
    ]
    if range_issues:
        lines += range_issues
    else:
        lines.append("None – all parameter ranges match paper values.")

    lines += [
        "",
        "## Anatomical model IDs",
        "- Model ID: `group::source_shape_id`, as selected for this analysis; source shape IDs are read from `info1` (LAE uses its `pwave_NNN` token).",
        f"- Unique group-scoped model IDs: **{df['model_id'].nunique() if 'model_id' in df else 0}**",
        f"- Model IDs represented in multiple groups (group count -> ID count): **{group_counts}**",
        f"- Model IDs appearing in multiple splits: **{len(overlap)}**",
        f"- Raw `info1` shape tokens: **{df['source_shape_id'].nunique() if 'source_shape_id' in df else 0} unique; group count -> token count: {source_shape_groups.value_counts().sort_index().to_dict()}**",
        f"- Directory `run_id` values retained for sensitivity analysis: **{df['run_id'].nunique()} unique; {len(runs_across_splits)} appear in multiple splits**.",
    ]
    if not runs_across_splits.empty:
        lines.append(f"- Directory run IDs spanning splits: **{runs_across_splits.to_dict()}**")
    if len(overlap) == 0:
        lines.append("All indexed model IDs appear in exactly one split.")
    else:
        lines.append("WARNING: model IDs appear in multiple splits:")
        lines.append(split_overlap_details.to_markdown())

    lines += ["", "## Rows not indexed against expected counts"]
    if underfilled_folders.empty:
        lines.append("None.")
    else:
        lines.append(underfilled_folders[["folder", "missing"]].to_markdown(index=False))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Mapping report written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()
    run_exp1(args.force_rebuild)
