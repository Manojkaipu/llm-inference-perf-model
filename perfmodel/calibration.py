"""Efficiency profiles available for each GPU, including measured calibrations."""
import json
import math
from pathlib import Path

from .hardware import GPUS, PEAK, REALISTIC, Efficiency

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _from_json(d: dict) -> Efficiency:
    d = dict(d)
    if d.get("overlap") is None:
        d["overlap"] = math.inf
    return Efficiency(**d)


def load_profiles(results_dir: Path = RESULTS) -> dict[str, dict[str, tuple[Efficiency, str]]]:
    """{gpu: {profile: (Efficiency, provenance)}}.

    Every GPU gets "peak" and "prior". A GPU with a validation run in
    results/calibrated_*_<gpu>_*.json also gets "calibrated" (the refined fit).
    """
    profiles = {g: {"peak": (PEAK, "datasheet peaks, no overhead (upper bound)"),
                    "prior": (REALISTIC[g], "hand-set efficiencies, not measured")}
                for g in GPUS}
    for path in sorted(Path(results_dir).glob("calibrated_*.json")):
        stem = path.stem.removeprefix("calibrated_")
        gpu = next((g for g in GPUS if f"_{g}_" in f"_{stem}_"), None)
        data = json.loads(path.read_text())
        if gpu and "refined" in data:
            profiles[gpu]["calibrated"] = (
                _from_json(data["refined"]),
                f"fitted to measured runs ({stem}); validated within ~9% decode / ~4% prefill "
                "held-out MAPE for that model and engine. Other models on this GPU are an extrapolation.")
    return profiles
