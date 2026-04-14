

# Metric Learning for Face Recognition

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: G25
- **Project ID**: 1

## 📝 Project Description
A brief paragraph (3-4 lines) that visually and concisely describes the project, the main implemented model, and the task addressed. 
*(Imagine this is the technical Abstract of your GitHub repo).*

> 📖 **Official Report**: For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**
Explain how the reader can install the environment to run your code.

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:**
Download CASIA-WebFace from [the official repository](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and place `train.rec`, `train.idx`, and `train.lst` in `data/casia-webface/`.

The dataset uses MXNet RecordIO format (`.rec`/`.idx` binary files). To extract images to disk, use the dedicated extraction environment (Python 3.7 with legacy mxnet support):

```bash
# Create extraction environment (one-time setup)
conda env create -f environment.extract.yml
conda activate dataset-extraction

# Extract .rec images to default directory (data/casia-webface/)
python src/datasets/extract_casia_rec.py

# Return to main environment
conda activate dl-project
```

After extraction, generate identity-disjoint train/val/test splits:

```bash
python src/datasets/make_split.py --data-root data/casia-webface --output data/splits/casia_identity_split_v1.json
```

### 2. Network Training
Provide the **exact commands** to start the training.

**Baseline Training:**
```bash
python -m src.training.train --config experiments/configs/base.yaml
```

**Improved Model Training:**
```bash
python -m src.training.train --config experiments/configs/triplet_hardmining.yaml
```

**On the cluster:**
```bash
bash cluster/train.sh
# or, to submit training + evaluation together
bash cluster/run_all.sh --config=experiments/configs/triplet_hardmining.yaml
```

### 3. Evaluation
Provide the commands to reproduce the numbers in your summary table.

```bash
python -m src.evaluation.evaluate --config experiments/configs/triplet_hardmining.yaml
```

---
