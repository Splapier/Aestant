"""DINOv2 + LinUCB image preference bandit.

A self-contained system that extracts dense DINOv2 features from a directory
of images (cached on disk with content fingerprints so images are never
re-processed), scores them with an online LinUCB bandit, shows the user the
top-2 candidates from a random batch at a time, and learns from each choice.
The user preference profile is saved after every choice and reloaded on
app restart.
"""

from image_bandit.feature_store import (
    DINOV2_MODEL,
    FEATURE_DIM,
    IMAGE_EXTENSIONS,
    FeatureStore,
    extract_dense_features,
    load_best_frame,
    pick_best_frame,
    scan_image_directory,
)
from image_bandit.linucb import LinUCBUser
from image_bandit.manual_inference import (
    ContentFeatureStore,
    InferenceResult,
    ScanResult,
    get_profile,
    run_manual_inference,
)
from image_bandit.preference_profile import (
    load_preference_profile,
    save_preference_profile,
)
from image_bandit.recommender import (
    LOSER_REWARD,
    WINNER_REWARD,
    BanditRecommender,
    Pair,
)

__all__ = [
    "DINOV2_MODEL",
    "FEATURE_DIM",
    "IMAGE_EXTENSIONS",
    "FeatureStore",
    "LinUCBUser",
    "BanditRecommender",
    "Pair",
    "WINNER_REWARD",
    "LOSER_REWARD",
    "ContentFeatureStore",
    "InferenceResult",
    "ScanResult",
    "extract_dense_features",
    "get_profile",
    "load_preference_profile",
    "run_manual_inference",
    "save_preference_profile",
    "scan_image_directory",
]
