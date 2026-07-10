"""Path + project-config bootstrap for the web layer.

The algorithm package lives in ``src/pickleball_phase2`` and is not pip-installed
(the test-suite adds ``src`` to ``sys.path`` the same way). This module is the
single place that wires that up, so every other webapp module can simply
``from pickleball_phase2 import ...``. Import it before anything that touches the
package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_project_config() -> Any:
    """Load the project's ``config.yaml`` via the existing typed ``Config``."""
    from pickleball_phase2.config import Config  # noqa: WPS433 (after path setup)

    return Config.load(REPO_ROOT / "config.yaml")
