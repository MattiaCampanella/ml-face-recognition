

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
Explain in 2 lines where to download the data from and in which folder it needs to reside (e.g., `data/raw/`).

### 2. Network Training
Provide the **exact commands** to start the training.

**Baseline Training:**
```bash
python -m src.training.train --config experiments/configs/baseline.yaml
```

**Improved Model Training:**
```bash
python -m src.training.train --config experiments/configs/model_v1.yaml
```

### 3. Evaluation
Provide the commands to reproduce the numbers in your summary table.

```bash
python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml
```

---
