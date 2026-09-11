# MeterSerial_VLM

A modular, model-independent OCR pipeline for extracting **meter serial numbers** and **IMEI** from digital electricity-meter images. Built for CPU and GPU deployment, with support for multiple OCR backends and rigorous evaluation.

---

## 📌 Project Overview

The pipeline extracts the following fields from meter images:

- **Meter Serial Number** (e.g., `U5772898`)
- **IMEI Number** (15-digit cellular module ID)
- **Date fields** (manufacturing, installation, etc.)
- **Meter ON/OFF status** (based on visible green backlight)

The system is **model-agnostic** — OCR backends (PaddleOCR, LightOnOCR, or future VLMs) can be swapped without rewriting the extraction or consolidation logic.

---

## 🎯 Core Design Principles

1. **Model Independence** — OCR backends are pluggable.
2. **Universal Extraction** — Same extractor works for PaddleOCR and LightOnOCR.
3. **Universal Consolidation** — Combines results across 7 image regions (1 full + 6 tiles).
4. **Safety-First Extraction** — Returns `NOT_FOUND` instead of guessing.
5. **Lazy Model Loading** — Models load only when needed.

---

## 🏗️ Architecture
Meter Image
│
▼
┌────────────────────┐
│ 1 Full + 6 Tiles │ ← 2×3 overlapping grid
└────────────────────┘
│
▼
┌────────────────────┐
│ OCR Backend │
│ ┌────────┬───────┐ │
│ │Paddle │Light │ │
│ │OCR │OnOCR │ │
│ └────────┴───────┘ │
└────────────────────┘
│
▼
Raw OCR Results
│
▼
┌────────────────────┐
│ Universal Extractor│
└────────────────────┘
│
▼
┌────────────────────┐
│ Universal Consolid.│
└────────────────────┘
│
▼
Final Structured Data

text

---

## 📂 Repository Structure
MeterSerial_VLM/
├── data/
│ ├── images/ # Downloaded meter images (gitignored)
│ └── test_10/ # 10-image benchmark set (gitignored)
│
├── models/
│ ├── base/
│ │ └── base_model.py
│ ├── backends/
│ │ ├── paddleocr/
│ │ │ └── paddleocr_backend.py
│ │ └── lightonocr/
│ │ └── lightonocr_backend.py
│ └── registry.py
│
├── scripts/
│ ├── download_dataset.py # Multithreaded image downloader
│ ├── test_paddleocr.py # Single-image test
│ ├── test_lightonocr.py # Single-image test
│ ├── evaluate_paddleocr.py # Benchmark evaluation
│ ├── analyze_results.py # Confusion matrix generator
│ ├── speed_test.py # Performance diagnostics
│ └── visualize_tiles.py # Tiling visualization
│
├── utils/
│ ├── ocr_extractor.py # Universal extractor
│ ├── ocr_consolidator.py # Universal consolidator
│ ├── lightonocr_extractor.py # LightOnOCR extractor wrapper
│ └── lightonocr_consolidator.py # LightOnOCR consolidator wrapper
│
├── configs/
├── prompts/
├── requirements-paddle.txt
├── requirements.txt
├── urls.txt # Image URLs
├── README.md
└── .gitignore

text

---

## ⚙️ Environment Setup

The project uses **two isolated Python environments** to prevent dependency conflicts.

### 🟢 Environment 1: PaddleOCR (`.paddlevenv`)

**Python Version:** 3.11  
**Purpose:** Fast, traditional OCR engine

```powershell
py -3.11 -m venv .paddlevenv
.\.paddlevenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddleocr==3.7.0 paddlex==3.7.2
Verify:

powershell
python -c "import paddle; print('Paddle:', paddle.__version__)"
🔵 Environment 2: LightOnOCR (.venv)
Python Version: 3.14 (or 3.11)
Purpose: Vision-Language Model for context-aware reading

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -U huggingface_hub qwen-vl-utils
hf download lightonai/LightOnOCR-2-1B --local-dir model_store/lightonocr
hf download Qwen/Qwen2-VL-2B-Instruct --local-dir model_store/qwen2-vl-2b
hf download OpenGVLab/InternVL2_5-4B --local-dir model_store/internvl2_5-4b
🚀 Usage
🖼️ Step 1: Download Images
Add image URLs to urls.txt, then:

powershell
python -m scripts.download_dataset
Downloads all images to data/images/ in parallel (~40 images/sec).

🧪 Step 2: Test on a Single Image
PaddleOCR:

powershell
$env:FLAGS_use_dnnl = "0"; $env:GLOG_minloglevel = "2"; python -m scripts.test_paddleocr data/images/sample.png
LightOnOCR:

powershell
python -m scripts.test_lightonocr data/images/sample.png
📊 Step 3: Run the Benchmark
Create data/test_10/ground_truth.csv:

csv
filename,true_serial,true_imei
image1.png,U5772898,861729077005025
Copy 10 images to data/test_10/.

Run:

powershell
$env:FLAGS_use_dnnl = "0"; $env:GLOG_minloglevel = "2"; python -m scripts.evaluate_paddleocr
python -m scripts.analyze_results
📊 Benchmark Results (10 Images)
Field	Accuracy	Precision	Recall	F1-Score	Char Accuracy
Serial Number	90.00%	90.00%	90.00%	90.00%	97.50%
IMEI	90.00%	100.00%	90.00%	94.74%	90.00%
Key Observations
✅ 100% IMEI Precision — Zero hallucination. Every IMEI output is correct.

✅ 97.5% Character Accuracy — Underlying OCR engine is nearly perfect.

⚠️ 1 Serial Truncation — U5772898 read as U57728 (tile boundary issue).

⚠️ 1 IMEI Miss — IMEI not visible in one image (correctly returned NOT_FOUND).

Industry Interpretation
Metric	Industry Benchmark	Status
Serial F1 ≥ 90%	Baseline automation	✅
IMEI Precision = 100%	Production-grade	✅
Char Accuracy ≥ 95%	High quality	✅
Estimated STP Rate	~90%	✅
🛠️ Key Implementations
1. Manual Image Resize (Speed Optimization)
Resizes images to 1600px max before OCR, reducing CPU inference from ~207s to ~15s per image.

2. Regex Patterns for Indian Meter Labels
Handles both IMEI NO and INET NO variants:

python
r'(?:INET NO|IMEI)\s*[:.]?\s*(\d{15})'
3. Fallback Serial Detection
Detects unlabeled serial numbers (e.g., standalone U5028045).

4. String-Safe CSV Reading
Prevents 15-digit IMEIs from being parsed as floats.

5. Multi-Threaded Download
Downloaded 1,998 images in 49 seconds (~40 images/sec).

⚠️ Known Limitations
Limitation	Impact	Mitigation
CPU Inference is Slow	~200s/image on Intel i5-8265U	Use Colab T4 GPU (~2s/image)
Tile Boundary Truncation	Serials may lose trailing chars	Increase tile_overlap to 0.45
Low-Res Resize	Small IMEIs lost	Increase max_size to 2400 on GPU
PaddlePaddle 3.3.1 Bug	OneDNN crash on CPU	Use PaddlePaddle 3.2.0 or FLAGS_use_dnnl=0
LightOnOCR Extremely Slow on CPU	~20 min/image	Use Colab T4 GPU only
🌐 GPU Deployment (Google Colab)
For large-scale processing, use Colab's T4 GPU for ~70x speedup.

python
!nvidia-smi
!pip install -q paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
!pip install -q paddleocr==3.7.0 paddlex==3.7.2
!git clone https://github.com/subreenabano/MeterSerial_VLM.git
%cd MeterSerial_VLM
Expected Performance
Environment	Per Image	1,998 Images
Local CPU (i5-8265U)	~200s	~110 hours
Colab T4 GPU	~2s	~60 minutes
📋 Roadmap
☑ PaddleOCR backend (CPU)
☑ LightOnOCR backend (CPU)
☑ Universal extractor & consolidator
☑ Multi-threaded image downloader
☑ Confusion matrix evaluation framework
☑ 10-image benchmark: 90% F1
□ Full 1,998-image batch run (Colab GPU)
□ ROI-based cropping (fixed regions)
□ ON/OFF classifier
□ LightOnOCR fallback for hard images
🧠 Development Guidelines
Keep Model-Specific Code in Backends
✅ models/backends/paddleocr/

✅ models/backends/lightonocr/

❌ Do NOT put model-specific logic in utils/ocr_extractor.py

Prefer NOT_FOUND Over Guessing
Correct value > NOT_FOUND > guessed value

A wrong identifier is worse than a missing one. This is critical for meter serials and IMEIs.

Preserve the Current Baseline
1 full image + 6 overlapping tiles = 7 regions

Do not change this strategy without evaluating it against the current approach.

🧪 Testing & Evaluation
The pipeline includes a rigorous evaluation framework:

Confusion Matrix: TP, TN, FP, FN per field

Precision & Recall: Identify hallucination vs miss patterns

F1-Score: Balanced accuracy metric

Character Accuracy: Even when wrong, how close was the model?

Run python -m scripts.analyze_results to generate the full report.

📞 Contact
Project: BCITS – Meter Serial Extraction
Developer: Subreena Bano
Status: Active Development

📄 License
Internal use – BCITS

text

**Save** the file (`Ctrl+S`).

---

## G3. Commit and push the README

```powershell
git add README.md
git commit -m "Add comprehensive project README with benchmark results and setup guide"
git push
