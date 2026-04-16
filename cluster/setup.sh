#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster DMI -- Face Recognition
#
# Uso (dal login node):
#   cd ~/ml-face-recognition
#   bash cluster/setup.sh
#
# Lo script rilancia se stesso dentro srun + Apptainer automaticamente.
# ============================================================================

# -- 0. Auto-rilancio dentro srun + Apptainer se siamo sul login node ----------
if [ -z "$APPTAINER_CONTAINER" ]; then
    echo "Login node rilevato -> rilancio inside srun + Apptainer..."
    ACCOUNT="${SLURM_ACCOUNT:-dl-course-q2}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
         --gres=gpu:1 --gres=shard:22000 --mem=48G --cpus-per-task=8 \
         apptainer run --nv /shared/sifs/latest.sif \
         bash "$0" "$@"
fi

set -e

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

mkdir -p logs experiments/runs experiments/checkpoints experiments/logs data

echo "=== Face Recognition Cluster Setup ==="
echo "Project root: $PROJ_DIR"
echo ""

# -- 1. Rilevamento Python e GPU -----------------------------------------------
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERRORE: Python non trovato!"
    exit 1
fi
echo "   Python: $($PY --version 2>&1)"

GPU_INFO=$($PY -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {name} (CC {cc[0]}.{cc[1]}, {vram:.1f} GB)')
else:
    print('  GPU: NESSUNA GPU rilevata')
" 2>/dev/null) || echo "  (PyTorch non ancora installato, GPU check dopo installazione)"

echo "$GPU_INFO"

# -- 2. Installazione dipendenze -----------------------------------------------
echo ""
echo "Installazione dipendenze..."
pip install --user -r cluster/requirements.txt
echo "   Dipendenze installate."

# -- 3. Verifica installazione -------------------------------------------------
echo ""
echo "Verifica installazione..."
$PY -c "
import torch, torchvision
print(f'  PyTorch:      {torch.__version__}')
print(f'  TorchVision:  {torchvision.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
try:
    import sklearn
    print(f'  scikit-learn: {sklearn.__version__}')
except ImportError:
    print(f'  scikit-learn: NON installato')
try:
    import yaml
    print(f'  PyYAML:       {yaml.__version__}')
except ImportError:
    print(f'  PyYAML:       NON installato')
try:
    import PIL
    print(f'  Pillow:       {PIL.__version__}')
except ImportError:
    print(f'  Pillow:       NON installato')
try:
    import tensorboard
    print(f'  TensorBoard:  {tensorboard.__version__}')
except ImportError:
    print(f'  TensorBoard:  NON installato')
"

# -- 4. Download e estrazione dataset ------------------------------------------
echo ""

# Controlla se le immagini estratte esistono già (una sottocartella con .jpg)
EXTRACTED=$(find data/casia-webface -name "*.jpg" 2>/dev/null | head -1)

if [ -n "$EXTRACTED" ]; then
    echo "Dataset già estratto: data/casia-webface (found)"
else
    # Download dataset da Kaggle
    if [ -f "data/casia-webface/train.rec" ]; then
        echo "Dataset .rec già scaricato, skip download."
    else
        echo "Download dataset da Kaggle..."
        $PY src/datasets/download_dataset.py
    fi

    # Estrazione immagini da .rec
    if [ -f "data/casia-webface/train.rec" ]; then
        echo "Installazione mxnet per estrazione..."
        pip install --user -r cluster/requirements_extract.txt
        echo "Estrazione immagini dal dataset .rec..."
        $PY -m src.datasets.extract_casia_rec --data-root data/casia-webface
    else
        echo "[WARN] train.rec non trovato, impossibile estrarre il dataset."
    fi
fi

if [ -f "data/splits/casia_identity_split_v1.json" ]; then
    echo "Split file: data/splits/casia_identity_split_v1.json (found)"
else
    echo "Split file: data/splits/casia_identity_split_v1.json (missing)"
    echo "  Generalo con:"
    echo "    python -m src.datasets.make_split --data-root data/casia-webface --output data/splits/casia_identity_split_v1.json --version v1 --overwrite"
fi

# -- 5. Riepilogo comandi ------------------------------------------------------
echo ""
echo "=== Setup completato! ==="
echo ""
echo "Prossimi passi:"
echo "  1. CONFIG=experiments/configs/base.yaml sbatch cluster/train.sh"
echo "  2. CONFIG=experiments/configs/base.yaml sbatch cluster/eval.sh"
echo "  3. Oppure pipeline completa: bash cluster/run_all.sh"
