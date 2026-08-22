"""Persistence of the LinUCB user preference profile across app restarts."""

from pathlib import Path

import numpy as np

from image_bandit.linucb import LinUCBUser


def save_preference_profile(profile: LinUCBUser, path) -> Path:
    """Save the bandit's A matrix, b vector, and hyperparameters to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        A=profile.A,
        b=profile.b,
        alpha=np.float64(profile.alpha),
        d=np.int64(profile.d),
        updates=np.int64(profile.updates),
    )
    return path


def load_preference_profile(
    path, expected_dim: int | None = None
) -> LinUCBUser | None:
    """Load a saved profile, or return None if absent or dimension-mismatched."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as data:
        d = int(data["d"])
        alpha = float(data["alpha"])
        if expected_dim is not None and d != expected_dim:
            return None
        profile = LinUCBUser(d, alpha=alpha)
        profile.A = data["A"].copy()
        profile.b = data["b"].copy()
        profile.updates = int(data["updates"]) if "updates" in data else 0
    return profile
