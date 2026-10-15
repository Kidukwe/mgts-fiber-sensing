"""Descriptive prequential forecast errors; no event accuracy without labels."""

import math
from collections import Counter


class BaselineMetrics:
    """Equal weight to profiles, then to valid zones within each profile.

    Both baselines are evaluated on identical scored median observations.
    The robust detector baseline freezes during suspected changes.
    """

    def __init__(self) -> None:
        self.profiles = 0
        self.zone_observations = 0
        self.errors = {"robust": (0.0, 0.0), "persistence": (0.0, 0.0)}
        self.status_counts: Counter = Counter()

    def update(self, scores: tuple[dict, ...]) -> None:
        self.status_counts.update(item["status"] for item in scores)
        paired = {name: [] for name in self.errors}
        for item in scores:
            if item["status"] != "scored" or item["kind"] != "heating":
                continue
            if item["persistence"] is None:
                continue
            errors = (item["value"] - item["baseline"], item["value"] - item["persistence"])
            if not all(math.isfinite(value) for value in errors):
                continue
            for name, error in zip(paired, errors):
                paired[name].append(abs(error))
        if paired["robust"]:
            self.profiles += 1
            self.zone_observations += len(paired["robust"])
            for name, values in paired.items():
                # Store profile MAE and profile RMS, rather than dependent zone rows.
                count = len(values)
                mae = math.fsum(value / count for value in values)
                rms = math.hypot(*(value / math.sqrt(count) for value in values))
                old_mae, old_rms = self.errors[name]
                weight = 1 / self.profiles
                self.errors[name] = (
                    old_mae * (1 - weight) + mae * weight,
                    math.hypot(old_rms * math.sqrt(1 - weight), rms * math.sqrt(weight)),
                )

    def report(self) -> dict:
        result = {
            "evaluation": "causal descriptive errors; no held-out event validation",
            "profiles": self.profiles, "zone_observations": self.zone_observations,
            "score_status_counts": dict(self.status_counts),
            "event_precision": None, "event_recall": None,
        }
        for name, (mae, rms) in self.errors.items():
            result[name] = {"mae": mae if self.profiles else None,
                            "rmse": rms if self.profiles else None}
        return result
