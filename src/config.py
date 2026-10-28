"""
config.py – centralised paths and constants.
All paths are absolute so every script can be run from any working directory.
"""
from pathlib import Path
import os

# ── Root directories ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # sts-gate/
SCRATCH_DIR  = Path(os.environ.get("ECG_SCRATCH_DIR", str(PROJECT_ROOT)))
CACHE_DIR    = Path(os.environ.get("ECG_CACHE_DIR", str(SCRATCH_DIR / "cache")))
RUNS_DIR     = Path(os.environ.get("ECG_RUNS_DIR", str(PROJECT_ROOT / "runs")))
LOGS_DIR     = PROJECT_ROOT / "logs"
CKPT_DIR     = PROJECT_ROOT / "ckpt"

# ── Dataset roots ─────────────────────────────────────────────────────────────
DATASETS_ROOT    = Path(os.environ.get("ECG_DATASETS_ROOT", str(PROJECT_ROOT / "data")))

MEDALCARE_ROOT   = Path(os.environ.get(
    "MEDALCARE_ROOT",
    str(DATASETS_ROOT / "MedalCare-XL" / "MedalCare-XL"),
))
ECG_BUNDLE_ROOT = (
    Path(os.environ["ECG_BUNDLE_ROOT"])
    if os.environ.get("ECG_BUNDLE_ROOT") else None
)
MEDALCARE_NOISE  = MEDALCARE_ROOT / "WP2_largeDataset_Noise"
MEDALCARE_PARAMS = MEDALCARE_ROOT / "WP2_largeDataset_ParameterFiles"

PTBXL_ROOT       = Path(os.environ.get(
    "PTBXL_ROOT",
    str(DATASETS_ROOT / "ptb-xl" / "1.0.3"),
))
PTBXL_RECORDS    = PTBXL_ROOT / "records500"
PTBXL_DB_CSV     = PTBXL_ROOT / "ptbxl_database.csv"
PTBXL_SCP_CSV    = PTBXL_ROOT / "scp_statements.csv"

ECG_FM_REPO      = Path(os.environ.get(
    "ECG_FM_REPO",
    str(DATASETS_ROOT / "ecg-fm-main" / "ecg-fm-main"),
))
FAIRSEQ_SIGNALS_REPO = Path(os.environ.get(
    "FAIRSEQ_SIGNALS_REPO",
    str(SCRATCH_DIR / "vendor" / "fairseq-signals-src"),
))
ECG_FM_CKPT      = Path(os.environ.get(
    "ECG_FM_CKPT",
    str(CKPT_DIR / "mimic_iv_ecg_physionet_pretrained.pt"),
))

# ── Cache sub-directories ─────────────────────────────────────────────────────
CACHE_WINDOWS    = CACHE_DIR / "windows"      # float32 .npy, shape (N,2,12,2500)
CACHE_EMBEDDINGS = CACHE_DIR / "embeddings"   # .npy per layer
CACHE_TOKENS     = CACHE_DIR / "tokens"       # float16 shards for SAE

INDEX_PARQUET    = CACHE_DIR / "index.parquet"
SCALE_CONST_NPY  = CACHE_DIR / "scale_const.npy"
SAE_WEIGHTS      = CACHE_DIR / "sae_weights.pt"
SAE_CTRL_WEIGHTS = CACHE_DIR / "sae_ctrl_weights.pt"

# ── Signal constants ──────────────────────────────────────────────────────────
SAMPLE_RATE  = 500          # Hz — ECG-FM was trained at 500 Hz
WIN_SAMPLES  = 2500         # 5 s x 500 Hz
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# ── MedalCare folder -> 8-class group mapping ─────────────────────────────────
MI_SUBFOLDERS = ["LAD_0.3","LAD_1.0","LCX_0.3_ant","LCX_0.3_post",
                 "LCX_1.0_ant","LCX_1.0_post","RCA_0.3","RCA_1.0"]

FOLDER_TO_GROUP = {
    "sinus":   "sinus",
    "lbbb":    "lbbb",
    "rbbb":    "rbbb",
    "avblock": "avblock",
    "lae":     "lae",
    "fam":     "fam",
    "iab":     "iab",
}
for _mi in MI_SUBFOLDERS:
    FOLDER_TO_GROUP[f"mi/{_mi}"] = "mi"

GROUP_LABELS = sorted(set(FOLDER_TO_GROUP.values()))   # 8 classes, alphabetical
GROUP_TO_INT = {g: i for i, g in enumerate(GROUP_LABELS)}

# Expected indexed ECG counts from the dataset layout.
EXPECTED_FOLDER_COUNTS = {
    "sinus": 1300,
    "lbbb": 1300,
    "rbbb": 1300,
    "avblock": 1300,
    "fam": 1300,
    "iab": 1300,
    "lae": 1300,
    "mi/LAD_0.3": 1300,
    "mi/LAD_1.0": 1300,
    "mi/LCX_0.3_ant": 700,
    "mi/LCX_0.3_post": 650,
    "mi/LCX_1.0_ant": 600,
    "mi/LCX_1.0_post": 650,
    "mi/RCA_0.3": 1300,
    "mi/RCA_1.0": 1300,
}
EXPECTED_TOTAL_ECGS = sum(EXPECTED_FOLDER_COUNTS.values())

# ── Test A settings ───────────────────────────────────────────────────────────
PROBE_LAYERS  = [4, 6, 8, 10, 12]   # transformer layers to extract
PROBE_C_GRID  = [0.01, 0.1, 1.0, 10.0]
PROBE_N_JOBS  = max(1, int(os.environ.get("ECG_PROBE_N_JOBS", "4")))
LOCKED_LAYER  = None   # set after Test A validation sweep; update before Test B

# ── Test B – SAE settings (starting values, not yet benchmarked) ──────────────
SAE_D_IN    = 768
SAE_D_SAE   = 6144    # 8x expansion
SAE_K       = 32      # topk kept activations per token (batch-wide)
SAE_LR      = 3e-4
SAE_BATCH   = 4096
SAE_EPOCHS  = 3       # passes over token shards

# ── Test C – association thresholds ───────────────────────────────────────────
ASSOC_Q_THR       = 0.05
ASSOC_RHO_THR     = 0.25
ASSOC_MIN_ATOMS   = 10     # pass rule: at least this many confirmed atoms
