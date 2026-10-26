"""
install.py – Install project dependencies into the active Python environment.

Required packages not in the existing venv:
  - wfdb         (PTB-XL loading)
  - neurokit2    (morphology features)
  - pyarrow      (parquet – already present as dependency of pandas)
  - fairseq-signals (ECG-FM model)  <- must be installed from source
  - ecg-transform (ECG-FM preprocessing)

Usage:
    python src/install.py [--no-fairseq] [--python PATH]
"""
import subprocess, sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
FAIRSEQ_SIGNALS_DIR = PROJECT_ROOT / "vendor" / "fairseq-signals-src"

SIMPLE_PACKAGES = [
    "wfdb",
    "neurokit2",
    "tabulate",        # for markdown tables in reports
]

def pip_install(packages: list, python: Path):
    cmd = [str(python), "-m", "pip", "install", "--quiet"] + packages
    print(f"Installing: {packages}")
    subprocess.run(cmd, check=True)


def install_fairseq_signals(python: Path):
    """
    Clone and install fairseq-signals from source.
    The ECG-FM repo requires the Jwoo5 fork.
    """
    repo_url = "https://github.com/Jwoo5/fairseq-signals.git"

    if not FAIRSEQ_SIGNALS_DIR.exists():
        FAIRSEQ_SIGNALS_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning fairseq-signals to {FAIRSEQ_SIGNALS_DIR}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(FAIRSEQ_SIGNALS_DIR)],
            capture_output=False
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args)

    print("Installing fairseq-signals from source...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "-e", str(FAIRSEQ_SIGNALS_DIR)],
        check=True,
    )


def check_package(name: str, python: Path) -> bool:
    result = subprocess.run(
        [str(python), "-c", f"import {name.replace('-','_')}"],
        capture_output=True
    )
    return result.returncode == 0


def run_install(no_fairseq: bool = False, python: Path = Path(sys.executable)):
    print("=" * 60)
    print("ECG Gate – Install Script")
    print("=" * 60)
    print(f"Python: {python}")

    if REQUIREMENTS.exists():
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)],
            check=True,
        )
    else:
        pip_install(SIMPLE_PACKAGES, python)

    # fairseq-signals (the model framework)
    if not no_fairseq:
        if not check_package("fairseq_signals", python):
            install_fairseq_signals(python)
        else:
            print("fairseq_signals already installed.")

    # Verify required project packages, including the model framework unless skipped.
    print("\n--- Verification ---")
    required = [
        "numpy", "pandas", "sklearn", "scipy", "matplotlib", "pyarrow",
        "wfdb", "neurokit2", "tabulate", "ecg_transform",
    ]
    if not no_fairseq:
        required.append("fairseq_signals")
    missing = []
    for pkg in required:
        ok = check_package(pkg, python)
        print(f"  {pkg}: {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(pkg)
    if missing:
        raise RuntimeError(f"Required packages failed verification: {missing}")

    print("\nInstall complete.")
    print("\nNext step: download the checkpoint.")
    print("  1. Go to https://huggingface.co/wanglab/ecg-fm")
    print("  2. Download mimic_iv_ecg_physionet_pretrained.pt")
    print("  3. Place it in ckpt/mimic_iv_ecg_physionet_pretrained.pt")
    print("\nThen run: python src/exp0_smoke.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fairseq", action="store_true",
                    help="Skip fairseq-signals installation")
    ap.add_argument("--python", default=sys.executable,
                    help="Python executable to install dependencies into")
    args = ap.parse_args()
    run_install(args.no_fairseq, Path(args.python))
