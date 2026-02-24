# DL4CV

kaggle datasets download rawsi18/military-assets-dataset-12-classes-yolo8-format
kaggle datasets download aalihhiader/military-camouflage-soldiers-dataset-mcs1k
https://universe.roboflow.com/robert-paulson-fncbw/military-vehicles-detection-qwfnc/dataset/2/download/yolov8

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124



# 🎯 Military Equipment Detection System
### Situational Awareness PoC — Real-time Detection & Alerting


---

## 📋 Project Overview

A proof-of-concept system for detecting military equipment in RGB video streams and triggering threat-level alerts in real time. The system identifies 7 classes of objects, classifies them by threat level, and logs all detections to a local database.

### Threat Level Mapping

| ID | Class | Threat | Alert Behavior |
|----|-------|--------|----------------|
| 0 | `tank` | 🔴 HIGH | Immediate alert |
| 1 | `apc_ifv` | 🔴 HIGH | Immediate alert |
| 2 | `armored_car` | 🔴 HIGH | Immediate alert |
| 3 | `artillery` | 🟡 MEDIUM | Alert after 3 consecutive frames |
| 4 | `logistics_truck` | 🟡 MEDIUM | Alert after 3 consecutive frames |
| 5 | `soldier` | 🟡 MEDIUM | Alert after 3 consecutive frames |
| 6 | `civilian_vehicle` | ⚪ NONE | Display only — no alert |

---

## 📁 Project Structure

```
DL4CV/
├── configs/
│   └── dataset.yaml              # Ultralytics training config
├── data/
│   ├── raw/                      # Original downloaded datasets — NEVER MODIFY
│   │   ├── dataset-paulson/
│   │   │   ├── train/
│   │   │   │   ├── images/
│   │   │   │   └── labels/
│   │   │   ├── valid/
│   │   │   │   ├── images/
│   │   │   │   └── labels/
│   │   │   └── test/
│   │   │       └── images/       # No labels — expected
│   │   └── dataset-RAW/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── processed/                # Remapped labels — output of remap_labels.py
│   │   ├── dataset-paulson/
│   │   └── dataset-RAW/
│   └── final/                    # Merged train/val/test — input to training
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── notebooks/                    # EDA only
├── runs/                         # Training outputs — auto-generated
├── src/
│   ├── data/
│   │   ├── remap_labels.py        # Step 1 — remap class IDs to unified schema
│   │   ├── merge_datasets.py      # Step 2 — merge and split into final/
│   │   ├── visual_verification.py # Step 3 — visually verify labels
│   │   ├── cap_classes.py         # Step 4 — fix class imbalance
│   │   ├── verify_dataset.py      # Step 5 — audit final dataset
│   │   └──last_sanity_check.py
│   ├── training/
│   │   └── train.py               # Model training script
│   └── inference/
│       ├── detector.py            # MilitaryDetector class
│       ├── alerting.py            # AlertSystem class
│       └── visualization.py       # Frame drawing utilities
├── main.py                        # Entry point — runs full pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### Prerequisites

- Python 
- Git
- NVIDIA GPU with CUDA drivers (recommended) or CPU fallback

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd DL4CV
```

### Step 2 — Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, verify CUDA is available after install:

```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))
```

If `torch.cuda.is_available()` returns `False`, reinstall PyTorch with the correct CUDA version for your driver:

```bash
# Check your CUDA version first
nvidia-smi

# Install matching PyTorch (example for CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** PyTorch does not always have builds for the very latest CUDA versions. If your CUDA is 13.x, install the cu124 build — it will run fine due to backward compatibility.

### Step 4 — Verify Installation

```bash
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import torch; print('PyTorch OK')"
```


---

## 🗂️ Datasets

| Dataset | Source | Classes Used | Split |
|---------|--------|-------------|-------|
| Military Vehicles (Paulson) | [Roboflow](https://universe.roboflow.com/robert-paulson-fncbw/military-vehicles-detection-qwfnc/dataset/1) | tank, apc_ifv, armored_car, logistics_truck, soldier, civilian_vehicle | train, valid |
| Military Assets (RAW) | [Kaggle](https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format) | tank, apc_ifv, artillery, logistics_truck, soldier, civilian_vehicle | train, val, test |

### Downloading Datasets

**Paulson (straight up):**
```bash
https://universe.roboflow.com/robert-paulson-fncbw/military-vehicles-detection-qwfnc/dataset/2/download/yolov8
```
rename to **dataset-paulson**


**RAW (Kaggle):**
```bash
# Place kaggle.json in ~/.kaggle/ first
kaggle datasets download rawsi18/military-assets-dataset-12-classes-yolo8-format
unzip military-assets-dataset-12-classes-yolo8-format.zip -d data/raw/dataset-RAW
```

---

## 🔄 Data Pipeline

Run scripts in this exact order from the **project root**:

```bash
# Step 1 — Remap class IDs to unified schema
remap_labels.py

# Step 2 — Merge datasets and create train/val/test splits
merge_datasets.py

# Step 3 — Visually verify labels look correct (spot check)
visual_verification.py

# Step 4 — Fix class imbalance (cap civilian_vehicle to 20,000)
cap_classes.py

# Step 5 — Final audit to confirm everything is clean
verify_dataset.py
```



### Final Dataset Statistics

| Split | Images | Total Annotations |
|-------|--------|-------------------|
| Train | ~28,600 | ~65,000 |
| Val | ~3,915 | ~32,540 |
| Test | ~3,917 | ~33,449 |

**Train class distribution after capping:**
```
tank                ~19,000  (29.3%)
soldier             ~19,333  (29.7%)
civilian_vehicle    ~20,000  (30.8%)  ← capped from 211,468
apc_ifv              ~3,524  ( 5.4%)
logistics_truck      ~2,242  ( 3.4%)
artillery              ~468  ( 0.7%)
armored_car            ~410  ( 0.6%)
```

---

## 🏋️ Training

> **Owner: Суходольський Дмитро**

Training uses a two-phase frozen/unfrozen approach to leverage COCO pretrained weights while adapting to the military domain.

```bash
python src/training/train.py
```



---

## ✅ Pre-Training Checklist

Before starting training confirm all of these:

- [ ] `python src/data/verify_dataset.py` passes with no errors
- [ ] `check_det_dataset("configs/dataset.yaml")` returns valid config
- [ ] Image/label counts match across all splits
- [ ] No class IDs outside 0–6 exist in any label file
- [ ] No corrupt images in final dataset
- [ ] `civilian_vehicle` capped to ~20,000 in train
- [ ] Empty label files removed from train
- [ ] GPU available: `torch.cuda.is_available()` returns `True`

---

## ⚠️ Rules for All Collaborators

1. **Never modify `data/raw/`** — it is the original source of truth
2. **Never change class IDs** mid-project — it invalidates all processed labels
3. **Always run from project root** — all paths are relative to `DL4CV/`
4. **Commit to Git after every completed task** — not at the end of the day
5. **If you find a remapping bug** — fix `remap_labels.py` and re-run the full pipeline, never manually edit label files
6. **Use `best.pt` not `last.pt`** for deployment

---

## 📊 Evaluation Metrics

The model is evaluated on the held-out test set. Key metrics to report:

- **mAP@0.5** — overall and per class
- **False Positive Rate** — critical for operational use
- **Inference FPS** — must be real-time (>25 FPS) on target hardware

---

## 🔧 Troubleshooting

**`torch.cuda.is_available()` returns False**
→ Reinstall PyTorch with correct CUDA index URL for your driver version

**`cv2.imshow` crashes with GUI error**
→ You have `opencv-python-headless` installed. Run:
```bash
pip uninstall opencv-python-headless opencv-python -y
pip install opencv-python
```

**Ultralytics can't find dataset.yaml**
→ Always run scripts from the project root (`DL4CV/`), not from inside `src/`

**Training crashes on corrupt image**
→ Run the corrupt image check script in `src/data/verify_dataset.py`