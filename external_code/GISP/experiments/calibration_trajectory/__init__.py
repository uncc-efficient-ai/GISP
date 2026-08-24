"""Post-hoc evaluation utilities for calibration-budget trajectories."""

from pathlib import Path
import sys


# ``utils.load_module`` loads this directory as a top-level package. Add the
# extension root so sibling packages remain importable independently of model loading.
_GISP_EXTENSION_ROOT = Path(__file__).resolve().parents[2]
if str(_GISP_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_GISP_EXTENSION_ROOT))

from .eval_checkpoint_trajectory import CheckpointTrajectoryEvaluator

__all__ = ["CheckpointTrajectoryEvaluator"]
