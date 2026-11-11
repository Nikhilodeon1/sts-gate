"""
provenance.py – Recompute every bookkeeping number quoted in the dossier from raw saved files.

Nothing here is estimated or hand-edited: set sizes come from runs/C_confirmed_atoms*.csv,
split/mesh/class counts come from cache/index.parquet, and scalar results come from runs/*.json.
Output: runs/provenance_report.json (+ printed summary).

Run:  python src/provenance.py
"""
import sys, json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

EXCL = ["run_S64", "run_S67"]
SUFFIX = "_sensitivity_without_S64_S67"


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def split_tables(index: pd.DataFrame) -> dict:
    out = {"n_ecgs": int(len(index)), "n_meshes": int(index["model_id"].nunique())}
    out["ecgs_per_split"] = index["split"].value_counts().to_dict()
    out["meshes_per_split"] = {s: int(index.loc[index["split"] == s, "model_id"].nunique())
                               for s in ["train", "validation", "test"]}
    cls = {}
    for g in sorted(index["group"].unique()):
        sub = index[index["group"] == g]
        cls[g] = {
            "ecgs": {s: int((sub["split"] == s).sum()) for s in ["train", "validation", "test"]},
            "meshes": {s: int(sub.loc[sub["split"] == s, "model_id"].nunique())
                       for s in ["train", "validation", "test"]},
        }
    out["per_class"] = cls
    out["class_mesh_pairs_test"] = int(sum(v["meshes"]["test"] for v in cls.values()))
    out["meshes_spanning_multiple_classes"] = int((index.groupby("model_id")["group"].nunique() > 1).sum())
    return out


def atom_sets(csv: Path) -> dict:
    df = pd.read_csv(csv)
    fam = {
        "APD": df["param"].str.startswith("v_APD"),
        "stim": df["param"].str.startswith("v_stim"),
        "isch": df["param"].str.startswith("v_isch"),
        "morph": df["param"].str.startswith("morph_"),
    }
    sets = {k: set(df.loc[m, "atom"].astype(int)) for k, m in fam.items()}
    out = {"n_pairs": int(len(df)), "n_unique_atoms": int(df["atom"].nunique()),
           "sizes": {k: len(v) for k, v in sets.items()}, "pairwise_intersections": {},
           "triple_intersections": {}, "unions": {}}
    for a, b in combinations(sets, 2):
        out["pairwise_intersections"][f"{a}&{b}"] = len(sets[a] & sets[b])
    for a, b, c in combinations(sets, 3):
        out["triple_intersections"][f"{a}&{b}&{c}"] = len(sets[a] & sets[b] & sets[c])
    out["unions"]["APD|stim"] = len(sets["APD"] | sets["stim"])
    out["unions"]["APD|stim|isch"] = len(sets["APD"] | sets["stim"] | sets["isch"])
    out["unions"]["APD|stim|morph"] = len(sets["APD"] | sets["stim"] | sets["morph"])
    out["unions"]["all"] = len(sets["APD"] | sets["stim"] | sets["isch"] | sets["morph"])
    out["only_isch"] = len(sets["isch"] - (sets["APD"] | sets["stim"] | sets["morph"]))
    out["per_param_unique_atoms"] = df.groupby("param")["atom"].nunique().to_dict()
    out["abs_rho_conf_by_param"] = {
        p: {"n_pairs": int(len(g)), "median_abs": float(g["rho_conf"].abs().median()),
            "max_abs": float(g["rho_conf"].abs().max())}
        for p, g in df.groupby("param")
    }
    return out


def main():
    rep = {}
    index = pd.read_parquet(C.INDEX_PARQUET) if C.INDEX_PARQUET.exists() else None
    if index is not None:
        rep["full_index"] = split_tables(index)
        sens = index.loc[~index["run_id"].isin(EXCL)].reset_index(drop=True)
        rep["sensitivity_index"] = split_tables(sens)
        removed = index.loc[index["run_id"].isin(EXCL)]
        rep["excluded_runs"] = {
            "n_ecgs": int(len(removed)),
            "n_meshes": int(removed["model_id"].nunique()),
            "per_class_ecgs": removed["group"].value_counts().to_dict(),
        }
        # Token bookkeeping implied by sae.extract_tokens defaults (MedalCare path)
        n_each_tr, n_each_held = 5000 // 8, 1500 // 8
        tr_pool = sens[sens["split"] == "train"]
        held_pool = sens[sens["split"].isin(["validation", "test"])]
        n_tr = int(sum(min(n_each_tr, (tr_pool["group"] == g).sum()) for g in C.GROUP_LABELS))
        n_held = int(sum(min(n_each_held, (held_pool["group"] == g).sum()) for g in C.GROUP_LABELS))
        rep["implied_token_extraction"] = {
            "train_ecgs": n_tr, "held_ecgs_from_validation_plus_test": n_held,
            "tokens_per_ecg": 312, "train_tokens": n_tr * 312, "held_tokens": n_held * 312,
            "note": "Held-out pool is validation+test (not validation only); counts assume sae.extract_tokens defaults.",
        }

    for tag, name in [("sensitivity", f"C_confirmed_atoms{SUFFIX}.csv")]:
        p = C.RUNS_DIR / name
        if p.exists():
            rep["confirmed_atom_sets"] = atom_sets(p)

    for key, name in [("A_sensitivity", f"A_results{SUFFIX}.json"), ("A_primary", "A_results.json")]:
        a = _load_json(C.RUNS_DIR / name)
        if a:
            rep[key] = {
                "n_index_records": a["n_index_records"],
                "test_auc_best_layer": a["test_auc"],
                "ctrl_val_aucs": a["ctrl_val_aucs"],
                "layer_test_macro_auc": {l: v["test"]["macro_auc"] for l, v in a["layer_results"].items()},
                "layer_val_macro_auc": {l: v["val"]["macro_auc"] for l, v in a["layer_results"].items()},
            }
    for key, name in [("B_pretrained", f"B_results{SUFFIX}.json"), ("B_ctrl", f"B_results_ctrl{SUFFIX}.json")]:
        b = _load_json(C.RUNS_DIR / name)
        if b:
            rep[key] = b["checks"]
    c = _load_json(C.RUNS_DIR / f"C_results{SUFFIX}.json")
    if c:
        rep["C_results"] = c
    pn = C.RUNS_DIR / f"C_perm_null{SUFFIX}.npy"
    if pn.exists():
        arr = np.load(pn)
        rep["C_perm_null_counts"] = {"n_perms": int(len(arr)), "min": int(arr.min()), "max": int(arr.max()),
                                      "n_unique_values": int(len(np.unique(arr)))}
    rep["sae_training_config"] = {"lr": C.SAE_LR, "batch": C.SAE_BATCH, "epochs": C.SAE_EPOCHS,
                                  "d_sae": C.SAE_D_SAE, "k": C.SAE_K}
    for name in ["mesh_generalization_audit.json", "causal_ablation.json", "token_l0_distribution.json",
                 "monosemanticity_audit.json", "sae_seed_stability.json"]:
        j = _load_json(C.RUNS_DIR / name)
        rep[f"file_present::{name}"] = j is not None

    out = C.RUNS_DIR / "provenance_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
