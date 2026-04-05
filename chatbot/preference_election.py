"""Preference Election functionality for comparing images via VLM.

This module provides tournament-style election logic for selecting the best
image based on user preferences. It handles:
- Loading user preference JSON
- Scanning images from input directory
- Tournament pairing with bye handling for odd counts
- Single comparison via VLM
- Full election workflow
- Tag generation for final candidates

Usage Example:
    >>> from chatbot.preference_election import run_full_election, load_preference_json
    >>> pref = load_preference_json()
    >>> result = run_full_election(pref, "lmstudio", {"endpoint_url": "http://localhost:1234"})
    >>> print(f"Winner: {result.winner_image}, Tags: {result.tags}")
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import numpy as np

from chatbot.image_modules.image_loading import (
    scan_input_directory,
    load_image_as_numpy,
    SUPPORTED_EXTENSIONS,
)
from chatbot.providers import get_provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREFERENCE_JSON_PATH = PROJECT_ROOT / "preference.json"
INPUT_DIR = PROJECT_ROOT / "input"


@dataclass
class PreferenceData:
    """Container for user preference data loaded from JSON.

    Attributes:
        confidence_score: Integer 0-100 indicating certainty of preferences.
        favored_features: List of tags the user likes.
        avoided_features: List of tags the user dislikes.
        text_hypothesis: User's written hypothesis about their preferences.

    Example:
        >>> pref = PreferenceData(75, ["blue_eyes"], ["dark"], "I like bright colors")
        >>> print(pref.confidence_score)
        75
    """

    confidence_score: int
    favored_features: list[str]
    avoided_features: list[str]
    text_hypothesis: str


@dataclass
class ElectionMatch:
    """Represents a single match between two images.

    Attributes:
        image_a: First image as numpy array.
        image_b: Second image as numpy array.
        winner: "A" or "B" indicating which image won.
        reason: Explanation for why the winner was chosen.
    """

    image_a: np.ndarray
    image_b: np.ndarray
    winner: str
    reason: str = ""


@dataclass
class ElectionResult:
    """Final result of a complete preference election.

    Attributes:
        winner_image: The winning image as numpy array.
        runner_up_image: The runner-up image as numpy array.
        winner_tags: List of tags generated for the winner image.
        runner_up_tags: List of tags generated for the runner-up image.
        explanation: Explanation of why the winner was chosen.
    """

    winner_image: np.ndarray
    runner_up_image: np.ndarray
    winner_tags: list[str] = field(default_factory=list)
    runner_up_tags: list[str] = field(default_factory=list)
    explanation: str = ""


def load_preference_json() -> PreferenceData:
    """Load preference data from the JSON file at project root.

    Returns:
        PreferenceData instance with loaded values.

    Raises:
        FileNotFoundError: If preference.json does not exist.

    Example:
        >>> pref = load_preference_json()
        >>> print(pref.text_hypothesis)
    """
    if not PREFERENCE_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Preference JSON not found at {PREFERENCE_JSON_PATH}. "
            "Call create_empty_preference_json() first."
        )

    with open(PREFERENCE_JSON_PATH, "r") as f:
        data = json.load(f)

    return PreferenceData(
        confidence_score=data.get("confidence_score", 0),
        favored_features=data.get("favored_features", []),
        avoided_features=data.get("avoided_features", []),
        text_hypothesis=data.get("text_hypothesis", ""),
    )


def create_empty_preference_json() -> PreferenceData:
    """Create an empty preference JSON template at project root.

    If the file already exists, it will not be overwritten.
    Returns the default preference data.

    Returns:
        PreferenceData with default empty values.

    Example:
        >>> pref = create_empty_preference_json()
        >>> print(pref.confidence_score)
        0
    """
    if not PREFERENCE_JSON_PATH.exists():
        default_data = {
            "confidence_score": 0,
            "favored_features": [],
            "avoided_features": [],
            "text_hypothesis": "",
        }
        with open(PREFERENCE_JSON_PATH, "w") as f:
            json.dump(default_data, f, indent=4)

    return PreferenceData(
        confidence_score=0,
        favored_features=[],
        avoided_features=[],
        text_hypothesis="",
    )


def scan_all_images() -> list[str]:
    """Scan the input directory for all available images.

    Unlike scan_input_directory which limits to 2, this returns all images.

    Returns:
        List of absolute file paths to images in input/ directory.

    Example:
        >>> paths = scan_all_images()
        >>> print(f"Found {len(paths)} images")
    """
    input_dir = INPUT_DIR.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        return []

    image_files = [
        f.absolute()
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    image_files.sort(key=lambda x: x.name.lower())
    return [str(f) for f in image_files]


def load_image_pair(paths: list[str]) -> list[np.ndarray]:
    """Load multiple images from paths as numpy arrays.

    Args:
        paths: List of image file paths.

    Returns:
        List of numpy arrays in RGB format.

    Example:
        >>> images = load_image_pair(["img1.png", "img2.png"])
        >>> print(len(images))
        2
    """
    images = []
    for path in paths:
        arr = load_image_as_numpy(path)
        if arr is not None:
            images.append(arr)
    return images


def pair_images_for_round(
    images: list[np.ndarray],
    randomize: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Pair images for a tournament round, handling odd counts with byes.

    Args:
        images: List of images to pair.
        randomize: Whether to shuffle before pairing (default True).

    Returns:
        Tuple of (pairs, byes) where:
        - pairs: List of (img_a, img_b) tuples
        - byes: List of images that get a bye this round

    Example:
        >>> imgs = [np.zeros((10,10,3)), np.zeros((10,10,3)), np.zeros((10,10,3))]
        >>> pairs, byes = pair_images_for_round(imgs)
        >>> len(pairs)  # 1 pair
        1
        >>> len(byes)   # 1 bye
        1
    """
    if randomize:
        working_images = images.copy()
        random.shuffle(working_images)
    else:
        working_images = images

    num_images = len(working_images)

    if num_images < 2:
        return [], working_images

    pairs = []
    byes = []

    if num_images % 2 == 0:
        for i in range(0, num_images, 2):
            pairs.append((working_images[i], working_images[i + 1]))
    else:
        for i in range(0, num_images - 1, 2):
            pairs.append((working_images[i], working_images[i + 1]))
        byes = [working_images[-1]]

    return pairs, byes


def build_comparison_prompt(
    image_a_path: str,
    image_b_path: str,
    preference: PreferenceData,
) -> list[dict[str, Any]]:
    """Build the prompt for comparing two images.

    Args:
        image_a_path: Path to first image.
        image_b_path: Path to second image.
        preference: User preference data.

    Returns:
        List of message dictionaries in multimodal format.

    Example:
        >>> pref = PreferenceData(75, ["blue"], ["dark"], "test")
        >>> msgs = build_comparison_prompt("a.png", "b.png", pref)
    """
    favored_str = (
        ", ".join(preference.favored_features)
        if preference.favored_features
        else "none"
    )
    avoided_str = (
        ", ".join(preference.avoided_features)
        if preference.avoided_features
        else "none"
    )

    system_prompt = """You are an expert at comparing images based on user preferences.
Respond ONLY with valid JSON, no other text. Use this exact format:
{"winner": "A" or "B", "reason": "brief explanation"}"""

    user_prompt = f"""Compare these two images for user preference.

User's preference hypothesis: {preference.text_hypothesis}
Features they like: {favored_str}
Features they dislike: {avoided_str}
Confidence score: {preference.confidence_score}/100

Examine both images and decide which one better matches the user's preferences.
Respond in JSON format: {{"winner": "A" or "B", "reason": "one sentence explanation"}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_comparison_response(response: str) -> dict[str, str]:
    """Parse the VLM's comparison response.

    Args:
        response: Raw response string from VLM.

    Returns:
        Dictionary with 'winner' and 'reason' keys.
        winner will be None if parsing fails.

    Example:
        >>> result = parse_comparison_response('{"winner": "A", "reason": "test"}')
        >>> result["winner"]
        'A'
    """
    import re

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            winner = data.get("winner", "").strip().upper()
            if winner in ("A", "B"):
                return {"winner": winner, "reason": data.get("reason", "")}
    except json.JSONDecodeError:
        pass

    return {"winner": None, "reason": f"Error parsing response: {response[:100]}"}


def run_single_comparison(
    image_a: np.ndarray,
    image_b: np.ndarray,
    preference: PreferenceData,
    provider_type: str,
    config: dict[str, str],
) -> ElectionMatch:
    """Run a single comparison between two images via VLM.

    Args:
        image_a: First image as numpy array.
        image_b: Second image as numpy array.
        preference: User preference data.
        provider_type: Provider type ("lmstudio" or "llamacpp").
        config: Provider configuration dict.

    Returns:
        ElectionMatch with winner information.

    Example:
        >>> pref = PreferenceData(50, [], [], "")
        >>> img1 = np.zeros((10, 10, 3))
        >>> img2 = np.ones((10, 10, 3)) * 255
        >>> result = run_single_comparison(img1, img2, pref, "lmstudio", {})
    """
    provider = get_provider(provider_type, config)

    messages = [
        {
            "role": "system",
            "content": """You are an expert at comparing images based on user preferences.
Respond ONLY with valid JSON, no other text.""",
        },
        {"role": "user", "content": _build_comparison_content(preference)},
    ]

    full_response = ""
    for chunk in provider.stream_chat(messages, [image_a, image_b]):
        if chunk:
            full_response += chunk

    parsed = parse_comparison_response(full_response)
    winner = parsed.get("winner", "A")
    reason = parsed.get("reason", "No reason provided")

    if winner == "A":
        return ElectionMatch(image_a, image_b, "A", reason)
    else:
        return ElectionMatch(image_a, image_b, "B", reason)


def _build_comparison_content(preference: PreferenceData) -> str:
    """Build the comparison prompt content string."""
    favored_str = (
        ", ".join(preference.favored_features)
        if preference.favored_features
        else "none"
    )
    avoided_str = (
        ", ".join(preference.avoided_features)
        if preference.avoided_features
        else "none"
    )

    return f"""Compare these two images for user preference.

User's preference hypothesis: {preference.text_hypothesis}
Features they like: {favored_str}
Features they dislike: {avoided_str}
Confidence score: {preference.confidence_score}/100

Image A: [first image]
Image B: [second image]

Examine both images and decide which one better matches the user's preferences.
Respond in JSON format only: {{"winner": "A" or "B", "reason": "one sentence explanation"}}"""


def run_tournament_round(
    images: list[np.ndarray],
    byes: list[np.ndarray],
    preference: PreferenceData,
    provider_type: str,
    config: dict[str, str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run a single tournament round.

    Args:
        images: List of images to compete in this round.
        byes: Images from previous round that get a bye this round.
        preference: User preference data.
        provider_type: Provider type string.
        config: Provider configuration dict.

    Returns:
        Tuple of (winners, remaining_byes) for next round.

    Example:
        >>> imgs = [np.zeros((10,10,3)), np.ones((10,10,3))*255]
        >>> wins, byes = run_tournament_round(imgs, [], pref, "lmstudio", {})
    """
    pairs, round_byes = pair_images_for_round(images, randomize=True)

    all_byes = byes + round_byes
    winners = []

    for img_a, img_b in pairs:
        match = run_single_comparison(img_a, img_b, preference, provider_type, config)
        if match.winner == "A":
            winners.append(match.image_a)
        else:
            winners.append(match.image_b)

    remaining_byes = list(all_byes)
    if len(remaining_byes) > 0 and len(winners) > 0:
        for bye_img in remaining_byes[:]:
            if not winners:
                break
            random_winner_idx = random.randint(0, len(winners) - 1)
            random_winner = winners[random_winner_idx]
            match = run_single_comparison(
                bye_img, random_winner, preference, provider_type, config
            )
            if match.winner == "A":
                winners[random_winner_idx] = bye_img
            remaining_byes.remove(bye_img)

    return winners, remaining_byes


def parse_tags_response(response: str) -> dict[str, list[str]]:
    """Parse tag generation response from VLM.

    Args:
        response: Raw response string from VLM.

    Returns:
        Dictionary with 'tags' key containing list of tags.

    Example:
        >>> result = parse_tags_response('{"tags": ["tag1", "tag2"]}')
        >>> result["tags"]
        ['tag1', 'tag2']
    """
    import re

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            tags = data.get("tags", [])
            if isinstance(tags, list):
                return {"tags": [str(t) for t in tags]}
    except json.JSONDecodeError:
        pass

    return {"tags": []}


def generate_tags_for_image(
    image: np.ndarray,
    provider_type: str,
    config: dict[str, str],
) -> list[str]:
    """Generate descriptive tags for an image via VLM.

    Args:
        image: Image as numpy array.
        provider_type: Provider type string.
        config: Provider configuration dict.

    Returns:
        List of generated tags.

    Example:
        >>> img = np.zeros((10, 10, 3))
        >>> tags = generate_tags_for_image(img, "lmstudio", {})
        >>> print(len(tags))
    """
    provider = get_provider(provider_type, config)

    messages = [
        {
            "role": "system",
            "content": "You are an expert at describing image features. Respond ONLY with valid JSON.",
        },
        {
            "role": "user",
            "content": """Analyze this image and describe its visual features as tags.
Generate 5-8 descriptive tags that capture: style, colors, subject matter, mood, composition.
Respond in JSON format: {"tags": ["tag1", "tag2", ...]}""",
        },
    ]

    full_response = ""
    for chunk in provider.stream_chat(messages, [image]):
        if chunk:
            full_response += chunk

    parsed = parse_tags_response(full_response)
    return parsed.get("tags", [])


def run_full_election(
    preference: PreferenceData,
    provider_type: str,
    config: dict[str, str],
) -> Generator[str, None, ElectionResult]:
    """Run a complete preference election tournament.

    Yields progress messages at each stage.

    Args:
        preference: User preference data.
        provider_type: Provider type string.
        config: Provider configuration dict.

    Yields:
        Status messages about election progress.

    Returns:
        ElectionResult with winner, runner-up, tags, and explanation.

    Example:
        >>> pref = PreferenceData(50, [], [], "")
        >>> for status in run_full_election(pref, "lmstudio", {}):
        ...     print(status)
        >>> result = run_full_election(pref, "lmstudio", {})
    """
    yield "Scanning images from input/..."
    image_paths = scan_all_images()
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images in input/ directory")

    yield f"Loading {len(image_paths)} images..."
    images = load_image_pair(image_paths)
    if len(images) < 2:
        raise ValueError("Could not load at least 2 images from input/")

    yield f"Starting tournament with {len(images)} images"
    byes = []
    round_num = 0

    while len(images) > 2:
        round_num += 1
        yield f"Round {round_num}: {len(images)} images competing"
        winners, new_byes = run_tournament_round(
            images, byes, preference, provider_type, config
        )
        images = winners
        byes = new_byes

    yield "Final round: determining winner..."
    if len(images) == 2:
        img_a, img_b = images[0], images[1]
    elif len(images) == 1 and len(byes) >= 1:
        img_a = images[0]
        img_b = byes[0]
        byes = byes[1:]
    else:
        raise RuntimeError("Unexpected state in tournament")

    final_match = run_single_comparison(img_a, img_b, preference, provider_type, config)

    if final_match.winner == "A":
        winner_image = final_match.image_a
        runner_up_image = final_match.image_b
    else:
        winner_image = final_match.image_b
        runner_up_image = final_match.image_a

    final_reason = final_match.reason

    yield "Generating tags for winner..."
    winner_tags = generate_tags_for_image(winner_image, provider_type, config)
    yield "Generating tags for runner-up..."
    runner_up_tags = generate_tags_for_image(runner_up_image, provider_type, config)

    explanation = f"Winner chosen because: {final_reason}"

    return ElectionResult(
        winner_image=winner_image,
        runner_up_image=runner_up_image,
        winner_tags=winner_tags,
        runner_up_tags=runner_up_tags,
        explanation=explanation,
    )


def get_all_tags(
    winner_tags: list[str],
    runner_up_tags: list[str],
) -> list[str]:
    """Combine and deduplicate tags from both final candidates.

    Args:
        winner_tags: Tags for winner image.
        runner_up_tags: Tags for runner-up image.

    Returns:
        Combined unique list of tags.

    Example:
        >>> tags = get_all_tags(["a", "b"], ["b", "c"])
        >>> sorted(tags)
        ['a', 'b', 'c']
    """
    all_tags = set(winner_tags) | set(runner_up_tags)
    return sorted(list(all_tags))


__all__ = [
    "PreferenceData",
    "ElectionMatch",
    "ElectionResult",
    "load_preference_json",
    "create_empty_preference_json",
    "scan_all_images",
    "load_image_pair",
    "pair_images_for_round",
    "build_comparison_prompt",
    "parse_comparison_response",
    "run_single_comparison",
    "run_tournament_round",
    "parse_tags_response",
    "generate_tags_for_image",
    "run_full_election",
    "get_all_tags",
]
