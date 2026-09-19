# MeterSerial-VLM — Colab Setup Guide

This guide covers everything needed to run the MeterSerial-VLM pipeline in
Google Colab, from a fresh runtime to a full extraction run: mounting Google
Drive, cloning this repo, installing PaddlePaddle + PaddleOCR (CPU or GPU),
and troubleshooting common failures.

## Contents

1. [One-shot Colab setup](#1-one-shot-colab-setup)
2. [Install PaddlePaddle + PaddleOCR — CPU build](#2-install-paddlepaddle--paddleocr--cpu-build)
3. [Install PaddlePaddle + PaddleOCR — GPU build](#3-install-paddlepaddle--paddleocr--gpu-build)
4. [Check whether a GPU is attached](#4-check-whether-a-gpu-is-attached)
5. [After every runtime restart](#5-after-every-runtime-restart)
6. [Running the pipeline](#6-running-the-pipeline)
7. [Troubleshooting](#7-troubleshooting)
8. [Appendix — file locations and configuration](#8-appendix--file-locations-and-configuration)
9. [Quick reference](#9-quick-reference)
10. [Changelog](#10-changelog)

---

## 1. One-shot Colab setup

**Run this cell first in every new Colab session.** It mounts Drive, clones
or pulls the repo, adds it to the Python path, and points the PaddleOCR model
cache at Drive so models persist between sessions.

The cell is idempotent — safe to run multiple times. If Drive is already
mounted it skips mounting; if the repo exists it pulls instead of cloning.

```python
# =========================================================
# Colab session setup — Drive + GitHub + PaddleOCR cache
# Run this cell FIRST after every runtime restart.
# =========================================================

import os
import sys
from pathlib import Path

# ---------- 1. Mount Google Drive ----------
from google.colab import drive
if not Path("/content/drive/MyDrive").exists():
    drive.mount('/content/drive')
else:
    print("✅ Drive already mounted")

# ---------- 2. Clone or pull repo ----------
REPO_URL = "https://github.com/subreenabano/MeterSerial_VLM.git"
REPO_DIR = "/content/MeterSerial_VLM"

if not Path(REPO_DIR).exists():
    print(f"⬇️  Cloning {REPO_URL} ...")
    !git clone {REPO_URL} {REPO_DIR}
else:
    print("🔄 Repo exists — pulling latest ...")
    !cd {REPO_DIR} && git pull

# ---------- 3. Move to repo + add to Python path ----------
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ---------- 4. Persist the PaddleOCR / PaddleX model cache on Drive ----------
# Must be set BEFORE paddleocr / paddlex is imported.
os.environ['PADDLE_PDX_CACHE_HOME'] = '/content/drive/MyDrive/paddlex_models'
os.makedirs(os.environ['PADDLE_PDX_CACHE_HOME'], exist_ok=True)

# ---------- 5. Summary ----------
print()
print("=" * 60)
print("SESSION SETUP COMPLETE")
print("=" * 60)
print(f"Working directory:  {os.getcwd()}")
print(f"Model cache:        {os.environ['PADDLE_PDX_CACHE_HOME']}")

# Quick check for PaddlePaddle + PaddleOCR
try:
    import paddle
    print(f"PaddlePaddle:       {paddle.__version__}  (CUDA: {paddle.device.is_compiled_with_cuda()})")
except ImportError:
    print("❌ PaddlePaddle not installed — see section 2 or 3")

try:
    import paddleocr
    print(f"PaddleOCR:          {paddleocr.__version__}")
except ImportError:
    print("❌ PaddleOCR not installed — see section 2 or 3")
```

**Expected output:**

```text
✅ Drive already mounted
🔄 Repo exists — pulling latest ...

============================================================
SESSION SETUP COMPLETE
============================================================
Working directory:  /content/MeterSerial_VLM
Model cache:        /content/drive/MyDrive/paddlex_models
PaddlePaddle:       3.2.0  (CUDA: False)
PaddleOCR:          3.7.0
```

---

## 2. Install PaddlePaddle + PaddleOCR — CPU build

Use this when **no GPU is attached** (see [section 4](#4-check-whether-a-gpu-is-attached) to check).

```python
!pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
!pip install paddleocr==3.7.0
```

- The CPU wheel is ~190 MB — expect ~2–3 minutes to download.
- After the install completes, **restart the runtime** (see [section 5](#5-after-every-runtime-restart)).

---

## 3. Install PaddlePaddle + PaddleOCR — GPU build

Use this when a **T4 or other GPU is attached** (see [section 4](#4-check-whether-a-gpu-is-attached) to check).

```python
!pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!pip install paddleocr==3.7.0
```

- The GPU wheel is ~1.9 GB — expect ~5–15 minutes depending on Colab's network speed.
- The install also pulls CUDA 12.6 libraries (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc.).
  This **downgrades the NVIDIA libraries Colab pre-installs for PyTorch**, which breaks PyTorch.

### ⚠️ Known side effect — Torch breaks after installing Paddle GPU

This project doesn't use PyTorch, but PaddleOCR 3.x tries to import
`transformers`, which in turn tries to import `torch`. If `torch` is broken,
the PaddleOCR import fails.

**Fix:** uninstall torch after installing Paddle GPU, then restart the runtime.

```python
!pip uninstall -y torch torchvision torchaudio
```

### ⚠️ Colab GPU quota

Colab's free tier limits GPU usage to a rolling ~12-hour window per day. Once
the quota is exhausted, **Runtime → Change runtime type → T4 GPU** shows
*"Cannot connect to GPU backend."*

Options:

| Option | Notes |
|---|---|
| Wait ~12–24 hours | Quota resets on the US Pacific calendar day |
| Click **Connect without GPU** | Proceed with the CPU install ([section 2](#2-install-paddlepaddle--paddleocr--cpu-build)) |
| Use a different Google account | Separate quota per account |
| Upgrade to Colab Pro (~$10/mo) | Removes the daily cap; adds L4 + A100 |
| Use Kaggle Notebooks | 30 hrs/week of T4/P100, separate quota |

---

## 4. Check whether a GPU is attached

**Method 1 — `nvidia-smi` (fastest):**

```python
!nvidia-smi
```

- GPU attached → prints a table with `Tesla T4` (or similar) and memory info.
- No GPU → prints `command not found` or `could not communicate with the NVIDIA driver`.

**Method 2 — Colab UI:**

Menu: **Runtime → Change runtime type**. The currently selected hardware
accelerator is shown in the dialog.

**Method 3 — Paddle's view:**

```python
import paddle
print("Compiled with CUDA: ", paddle.device.is_compiled_with_cuda())
print("CUDA device count:  ", paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0)
print("Current device:     ", paddle.device.get_device())
```

| Scenario | Compiled with CUDA | Device count | Current device |
|---|---|---|---|
| GPU attached + GPU wheel | `True` | 1 | `gpu:0` |
| CPU-only runtime + CPU wheel | `False` | 0 | `cpu` |
| CPU-only runtime + GPU wheel | `True` | 0 | `cpu` |

**Rule of thumb:**

- `nvidia-smi` succeeds **and** `device_count() > 0` → the GPU is truly available.
- If `nvidia-smi` fails, install the **CPU** wheel ([section 2](#2-install-paddlepaddle--paddleocr--cpu-build)).
  Don't download the 1.9 GB GPU wheel for a CPU-only runtime.

---

## 5. After every runtime restart

Colab resets the local filesystem on every runtime restart or hardware-type
change. Drive and pip packages follow different persistence rules:

| Resource | Persists across restart? |
|---|---|
| Files on Drive (`/content/drive/MyDrive/...`) | ✅ Yes |
| Files on local disk (`/content/...`) | ❌ No |
| pip-installed packages (same runtime type) | ✅ Yes |
| pip-installed packages (runtime type changed) | ❌ No |
| Model cache at `/content/drive/MyDrive/paddlex_models` | ✅ Yes |

### The two-step restart routine

1. **Run the setup cell** ([section 1](#1-one-shot-colab-setup)). It remounts
   Drive, clones/pulls the repo, sets the model cache path, and reports
   whether Paddle is installed.
2. **If Paddle is missing** → run [section 2](#2-install-paddlepaddle--paddleocr--cpu-build)
   (CPU) or [section 3](#3-install-paddlepaddle--paddleocr--gpu-build) (GPU),
   restart the runtime, then run the setup cell again.

### Same-runtime-type restarts (no reinstall needed)

If you only pressed **Runtime → Restart session** without changing the
hardware type, pip packages survive. Just run the setup cell.

### Hardware-type changes (reinstall required)

Switching between CPU / T4 / etc. wipes pip packages. Run the install cell
for the new hardware type again.

---

## 6. Running the pipeline

Once setup and install are done, verify that the backend loads correctly:

```python
import sys
sys.path.insert(0, "/content/MeterSerial_VLM")
import os
os.chdir("/content/MeterSerial_VLM")

# Force a fresh reload if any cached modules exist
for m in list(sys.modules.keys()):
    if m.startswith("models.backends.paddleocr"):
        del sys.modules[m]

from models.backends.paddleocr.paddleocr_backend import PaddleOCRBackend

backend = PaddleOCRBackend()
backend.load()
```

**Expected output:**

```text
[PaddleOCRBackend] Loaded on device: gpu   (or "cpu")
```

The backend **auto-detects the GPU at load time** — no manual switching required.

### Quick pipeline test (one image)

```python
from utils.ocr_extractor import UniversalOCRExtractor
from utils.ocr_consolidator import UniversalOCRConsolidator
from pathlib import Path
import requests

# Download one image
url = "https://storage.googleapis.com/bcits_hescom/_00000000010410028_pendnewmtrimg_newmtrphoto_10_10_2025_02_04_47.png"
img_bytes = requests.get(url, timeout=30).content
img_path = Path("/content/test_image.png")
img_path.write_bytes(img_bytes)

# Run pipeline
processed = backend.preprocess(img_path)
raw       = backend.predict(processed)
final     = backend.postprocess(raw)

extractor    = UniversalOCRExtractor()
consolidator = UniversalOCRConsolidator()

region_results = {}
for r in final["regions"]:
    region_results[r["region"]] = extractor.extract(r["raw_output"] or "")

result = consolidator.consolidate(region_results)
print("Serial:", result["serial_number"])
print("IMEI:  ", result["imei"])
```

**Expected output:**

```text
Serial: U5775686
IMEI:   861919082796862
```

---

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'paddle'`

- Paddle hasn't been installed yet, **or**
- you switched runtime types after installing.

**Fix:** run [section 2](#2-install-paddlepaddle--paddleocr--cpu-build) (CPU)
or [section 3](#3-install-paddlepaddle--paddleocr--gpu-build) (GPU), then
restart and re-run the setup cell.

### `ImportError: libtorch_cuda.so: undefined symbol: ncclCommShrink`

Torch was broken by the Paddle GPU install.

**Fix:** `!pip uninstall -y torch torchvision torchaudio`, then restart the runtime.

### `RuntimeError: PDX has already been initialized`

PaddleX was already initialized once in this kernel.

**Fix:** **Runtime → Restart session**.

### `Cannot connect to GPU backend`

Colab's free-tier GPU quota is exhausted.

**Fix:** wait ~12–24 hours, use a different Google account, or upgrade to
Colab Pro. See the quota options in [section 3](#-colab-gpu-quota).

### Slow inference (~100 s per image)

You're on CPU. A GPU does ~15–25 s per image.

**Fix:** confirm with `!nvidia-smi`. If there's no GPU, consider the GPU
install ([section 3](#3-install-paddlepaddle--paddleocr--gpu-build)) once
quota resets.

### Model files re-downloading on every restart

The cache location wasn't set, or was set to a non-Drive path, or was set
*after* PaddleOCR was imported.

**Fix:** run the setup cell ([section 1](#1-one-shot-colab-setup)) **before**
importing `paddleocr`. To check that the cache is being used, list the folder
after a first run:

```python
!ls /content/drive/MyDrive/paddlex_models
```

### `git pull` fails with local modifications

Colab shouldn't leave local edits, but if it does:

```python
!cd /content/MeterSerial_VLM && git stash && git pull
```

---

## 8. Appendix — file locations and configuration

### File locations

| Path | Purpose |
|---|---|
| `/content/MeterSerial_VLM` | Repo working directory |
| `/content/drive/MyDrive/A1_CLEAN.csv` | Input: 870 clean rows |
| `/content/drive/MyDrive/A1_SUSPECT_after_fix.csv` | Input: 192 suspect rows |
| `/content/drive/MyDrive/A1_SUSPECT_batch_results.csv` | Output from prior batch run |
| `/content/drive/MyDrive/paddlex_models` | Persistent PaddleOCR / PaddleX model cache |
| `/content/url_test_cache` | Downloaded test images (ephemeral) |

### Key source files

| File | Role |
|---|---|
| `models/backends/paddleocr/paddleocr_backend.py` | Runs OCR on 7 regions; auto-detects GPU/CPU |
| `models/registry.py` | Backend registration |
| `utils/ocr_extractor.py` | Pattern-first serial + IMEI extraction |
| `utils/ocr_consolidator.py` | Cross-region voting |
| `scripts/evaluate_paddleocr.py` | Batch driver *(to be updated to use per-region extraction)* |

### Environment variables

| Variable | Value | Purpose |
|---|---|---|
| `PADDLE_PDX_CACHE_HOME` | `/content/drive/MyDrive/paddlex_models` | Where PaddleX stores downloaded models, so they survive restarts. Set before importing `paddleocr`. |
| `PADDLE_PDX_MODEL_SOURCE` | *(optional)* `BOS` | Which server models are downloaded from. Defaults to HuggingFace; set to `BOS` if HuggingFace is unreachable. **This selects a download source, not a folder path.** |

> **Note:** The CPU-session and GPU-session models are different sets, but
> both live in the same Drive cache, so each is downloaded only once.

---

## 9. Quick reference

For sessions where Paddle is **already installed** and you just need to
reconnect Drive and the repo:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys
if not os.path.exists("/content/MeterSerial_VLM"):
    !git clone https://github.com/subreenabano/MeterSerial_VLM.git /content/MeterSerial_VLM
else:
    !cd /content/MeterSerial_VLM && git pull

os.chdir("/content/MeterSerial_VLM")
sys.path.insert(0, "/content/MeterSerial_VLM")

os.environ['PADDLE_PDX_CACHE_HOME'] = '/content/drive/MyDrive/paddlex_models'
os.makedirs(os.environ['PADDLE_PDX_CACHE_HOME'], exist_ok=True)
print("✅ Ready")
```

---

## 10. Changelog

- **v1** — Initial setup guide covering Drive, GitHub, CPU/GPU installs, GPU
  detection, restart procedures, and troubleshooting.
- **v1.1** — Reformatted as a repo README; model-cache setup now uses
  `PADDLE_PDX_CACHE_HOME` (previously `PADDLE_PDX_MODEL_SOURCE` was set to a
  folder path).
