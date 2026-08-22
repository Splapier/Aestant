"""LinUCB contextual bandit for learning user image preferences.

Port of LinUCBUser from dino.py: a closed-form linear preference model
(A, b) updated online with scalar rewards - no backpropagation needed.
"""

import numpy as np


class LinUCBUser:
    """Linear UCB bandit over dense image feature vectors."""

    def __init__(self, feature_dim: int, alpha: float = 0.1):
        self.alpha = alpha
        self.d = feature_dim
        # Initialize identity matrix A and zero vector b
        self.A = np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)
        self.updates = 0

    def _theta_and_inverse(self) -> tuple[np.ndarray, np.ndarray]:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        return theta, A_inv

    def score_images(self, image_vectors) -> np.ndarray:
        """Score a batch of feature vectors (one matrix inverse total)."""
        vectors = np.atleast_2d(np.asarray(image_vectors, dtype=np.float64))
        theta, A_inv = self._theta_and_inverse()
        # Expected baseline score
        expected_reward = vectors @ theta
        # Exploration bonus (Upper Confidence Bound)
        cb = self.alpha * np.sqrt(
            np.einsum("ij,jk,ik->i", vectors, A_inv, vectors)
        )
        return expected_reward + cb

    def score_image(self, image_vector) -> float:
        """Score a single feature vector (same math as dino.py)."""
        return float(self.score_images(image_vector)[0])

    def update_preferences(self, image_vector, reward: float) -> None:
        """Instant online update - no backpropagation needed."""
        v = np.asarray(image_vector, dtype=np.float64).reshape(-1)
        self.A += np.outer(v, v)
        self.b += reward * v
        self.updates += 1
