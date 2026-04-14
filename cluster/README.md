# Cluster scripts

Utility scripts for running the face-recognition baseline on a SLURM cluster.

The workflow is centered on the real layout of this repository:

- training data: `data/casia-webface`
- identity split: `data/splits/casia_identity_split_v1.json`
- configs: `experiments/configs/base.yaml`
- training outputs: `experiments/runs/<run_name>/`

Typical usage:

```bash
bash cluster/setup.sh
bash cluster/train.sh
bash cluster/eval.sh
```

`cluster/run_all.sh` submits training and then evaluation with a SLURM dependency.