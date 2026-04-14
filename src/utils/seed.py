from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	try:
		import numpy as np

		np.random.seed(seed)
	except ImportError:
		pass

	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	if deterministic:
		torch.use_deterministic_algorithms(True, warn_only=True)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = benchmark
