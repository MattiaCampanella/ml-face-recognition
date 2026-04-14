from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
	import yaml
except ImportError as exc:  # pragma: no cover - import guard
	raise ImportError(
		"PyYAML is required to load experiment configs. Install the 'pyyaml' package."
	) from exc


@dataclass(frozen=True)
class LoadedConfig:
	path: Path
	data: dict[str, Any]


def repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: str | Path) -> LoadedConfig:
	path = Path(config_path)
	if not path.is_absolute():
		path = repo_root() / path

	with path.open("r", encoding="utf-8") as handle:
		data = yaml.safe_load(handle)

	if not isinstance(data, dict):
		raise ValueError(f"Config file must contain a mapping, got {type(data).__name__}.")

	return LoadedConfig(path=path, data=data)


def resolve_repo_path(path_value: str | Path) -> Path:
	path = Path(path_value)
	if path.is_absolute():
		return path
	return repo_root() / path


def ensure_dir(path_value: str | Path) -> Path:
	path = resolve_repo_path(path_value)
	path.mkdir(parents=True, exist_ok=True)
	return path


def make_run_name(config: dict[str, Any]) -> str:
	project = config.get("project", {})
	output = config.get("output", {})
	naming = output.get("naming", {})

	experiment = project.get("experiment", "run")
	include_experiment = bool(naming.get("include_experiment", True))
	use_timestamp = bool(naming.get("use_timestamp", True))

	parts: list[str] = []
	if include_experiment and experiment:
		parts.append(str(experiment))
	if use_timestamp:
		parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))

	return "-".join(parts) if parts else "run"


def save_json(path_value: str | Path, payload: Any) -> Path:
	path = resolve_repo_path(path_value)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2)
		handle.write("\n")
	return path
