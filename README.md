# Aestant

A DINOv2 + LinUCB **image preference learning** system. A Gradio app scores images from a directory with a contextual bandit, shows you the top-2 candidates from a random batch, and learns your preferences from each choice (+1.0 / -1.0). Includes a read-only manual-inference mode and a set of CLI/GUI tools for finding and removing duplicate images.

## Features

- **Dense feature extraction** — 1152-d DINOv2 (`facebook/dinov2-small`) vectors, mean-pooled from transformer layers 3/7/11 and L2-normalized
- **Content-fingerprinted feature cache** — images are keyed by relative path and SHA-256, so already-processed images are never re-extracted (cache in `data/bandit/features/`)
- **GIF support** — animated images (`.gif`) are represented by their most detailed frame, so blank or transition frames do not dominate the features
- **LinUCB bandit** — closed-form linear preference model (A, b) updated online, no backpropagation; profile persisted to `data/bandit/preference_profile.npz` and reloaded on restart
- **Interactive preference loop** — pick a winner, skip a pair, or delete an image (removed from disk, slot refilled from the batch)
- **Manual inference** — score every image in `input/` with the learned profile and show the top image + top-5 scores; only reads the profile, never updates it
- **Duplicate image tools** — parallel pHash-based duplicate finders/deleters plus a tkinter review GUI
- **GPU acceleration** — DINOv2 runs on CUDA when available, CPU otherwise

## Project Structure

```
Aestant/
├── bandit_app.py                     # Entry point: launches the Gradio app on port 7861
├── image_bandit/                     # DINOv2 + LinUCB preference bandit
│   ├── __init__.py                   # Package exports
│   ├── app.py                        # Gradio UI (pair loop + manual inference section)
│   ├── feature_store.py              # DINOv2 extraction + path-keyed on-disk feature cache
│   ├── linucb.py                     # LinUCBUser (A/b matrices, scoring, online updates)
│   ├── preference_profile.py         # Save/load the profile as .npz
│   ├── recommender.py                # Batch draw → score → top-2 → choose/skip/delete loop
│   └── manual_inference.py           # Content-hash-keyed cache + read-only scoring of input/
├── duplicates_tools/                 # Duplicate image handling
│   ├── hash_manager.py               # Shared pHash cache (image_hashes.json), used by all tools
│   ├── duplicates_finder.py          # Find near-duplicate groups in one directory → JSON
│   ├── one_to_one.py                 # Find duplicates and auto-delete (keeps largest file)
│   ├── dir_to_dir.py                 # Delete files in a target dir that match a reference dir
│   ├── duplicates_reviewer.py        # tkinter GUI to review groups and pick keepers
│   └── dirdir_file_clash.py          # Rename files in dir A that clash with names in dir B
├── tests/                            # Unit tests (torch-free, fake extractor injected)
│   ├── test_image_bandit.py
│   └── test_manual_inference.py
├── images/                           # Bandit image pool (gitignored)
├── input/                            # Images for manual inference (gitignored)
└── data/bandit/                      # Feature caches + preference profile (gitignored)
```

## Prerequisites

- Python 3.12 or higher
- `pip`
- (Optional) A CUDA-capable GPU — DINOv2 feature extraction uses it automatically when available

## Installation

```bash
# From the repository root
pip install gradio numpy pillow torch transformers imagehash

# For running the tests
pip install pytest
```

The DINOv2 model (`facebook/dinov2-small`) is downloaded from Hugging Face automatically on first feature extraction and cached by `transformers`.

## Usage

### Starting the Application

```bash
python bandit_app.py
```

The app launches at `http://127.0.0.1:7861`.

Put your images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.gif`, top level of the directory) in `images/`, then:

1. On load, the app scans `images/` and computes dense features for any new/changed images (cached ones are skipped).
2. A random batch (default 8) is drawn and scored with the LinUCB profile; the **top-2 candidates** are shown.
3. **Pick a winner** — the profile is updated (+1.0 winner / -1.0 loser), saved, and the next pair from the batch is shown. When the batch is exhausted, a fresh batch is drawn.
4. **Skip** — discards the pair (no preference update) and shows the next top-2 of the batch.
5. **Delete** — removes the image file from disk and the feature store, keeps the other image in place, and refills the empty slot with the next-highest-scoring batch image.
6. **Scan** — re-scans `images/` for new/changed/removed images at any time.

The status line reports scan stats (scanned / new / cached / reprocessed / removed) and loop stats (batch remaining, pairs shown, skipped, deleted, preference updates).

### Manual Inference

The "🔍 Manual Inference" section scores every image in `input/` with the **current learned profile** and displays the top image with its score plus the top-5 scores. It runs automatically on load and via the Run button.

- It only **reads** the preference profile — it never updates it, so it cannot interfere with the bandit loop.
- Its feature cache (`data/bandit/input_features/`) is keyed by the **SHA-256 of the file contents**, not the path: moved, renamed, or re-added images are recognized by content and never re-extracted, and scans never prune entries.

### Resetting State

Everything learned and cached lives in `data/bandit/` (gitignored). Delete it to start fresh:

```bash
rm -rf data/bandit
```

## Duplicate Image Tools

All tools are run from the repository root as modules. They share a pHash cache (`image_hashes.json` in the current directory, managed by `hash_manager.py`) so unchanged images are hashed only once. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`.

| Tool | Purpose |
|------|---------|
| `duplicates_finder.py` | Find near-duplicate groups in one directory, write them to JSON |
| `one_to_one.py` | Find duplicates and auto-delete (keeps the largest file, shortest path) |
| `dir_to_dir.py` | Delete files in a target directory that match files in a reference directory |
| `duplicates_reviewer.py` | tkinter GUI to review groups, pick keepers, whitelist groups |
| `dirdir_file_clash.py` | Rename files in dir A whose names clash with files in dir B |

```bash
# Find near-duplicates in a directory (threshold = max pHash bit distance, default 5)
python -m duplicates_tools.duplicates_finder images/ -t 5 -o duplicates.json

# Strict dedup: delete exact pHash duplicates, keeping the largest file of each group
python -m duplicates_tools.one_to_one images/ -t 0

# Delete files in target/ that also exist in reference/ (preview first with --dry-run)
python -m duplicates_tools.dir_to_dir reference/ target/ -t 0 --dry-run

# GUI review of a duplicates.json file (moves non-kept files to deleted_duplicates/)
python -m duplicates_tools.duplicates_reviewer

# Rename name clashes between two directories (edit directory_a / directory_b at the bottom first)
python -m duplicates_tools.dirdir_file_clash
```

Notes:

- `duplicates_reviewer.py` requires a display (tkinter) and persists ignored groups in `whitelist_duplicates.json`.
- The delete tools remove files for real — use `--dry-run` (where supported) or `duplicates_finder.py` + the reviewer first.
- Animated GIFs are hashed (and shown in the reviewer) using their most detailed frame — the same frame the bandit app uses for features.

## Development

### Running the Tests

```bash
python -m pytest tests
```

58 tests, all torch-free: DINOv2 is never loaded because the feature stores accept an injectable fake extractor.

### How the Bandit Loop Works

`BanditRecommender` (`image_bandit/recommender.py`) drives the loop:

1. `next_pair()` draws a random batch from the `FeatureStore` (if the current one is exhausted), scores every batch member with `LinUCBUser.score_images()` (expected reward + UCB exploration bonus), and returns the top-2 as a `Pair` (higher score on the left).
2. `choose(side)` applies `update_preferences(+1.0)` to the winner and `update_preferences(-1.0)` to the loser, then saves the profile.
3. `skip()` discards the pending pair without updating the profile.
4. `delete_image(side)` unlinks the file, removes it from the store, and refills the slot with the next-highest-scoring batch image (a fresh batch is drawn if the current one is exhausted).

`LinUCBUser` (`image_bandit/linucb.py`) keeps the A matrix and b vector in memory and updates them in closed form — `A += v·vᵀ`, `b += r·v` — so each preference is applied instantly.

## License

This project is licensed under the terms specified in the LICENSE file.
