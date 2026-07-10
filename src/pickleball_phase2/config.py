"""Typed access to config.yaml. No placeholders in this file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


class Config:
    """Dot-access wrapper: cfg.get('fusion.dispute_threshold_ft')."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "Config":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f), source=path)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not None:
                    return default
                raise KeyError(f"config key not found: {dotted_key}")
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.get(key)
