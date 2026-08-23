import os
import json
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

try:
    from PIL import Image
    import imagehash
    import numpy as np
except ImportError as e:
    print("[-] Missing required libraries.")
    raise e

# --- Configuration ---
DB_FILENAME = "image_hashes.json"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

def _compute_single_hash(path):
    """
    Worker function.
    Handles the Pillow warning by converting modes securely.
    """
    try:
        # Open image
        image = Image.open(path)

        # FIX: Handle "Palette images with Transparency" warning
        # Convert P/LA/PA modes with transparency to RGBA first
        if image.mode in ('P', 'PA', 'LA') or (image.mode == 'L' and 'transparency' in image.info):
            image = image.convert('RGBA')

        # Compute pHash
        hash_obj = imagehash.phash(image)

        # Return hex string for JSON storage
        return path, str(hash_obj)
    except Exception as e:
        # print(f"Error hashing {path}: {e}") # Optional debugging
        return path, None

class HashManager:
    def __init__(self, db_path=DB_FILENAME):
        self.db_path = db_path
        self.cache = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print("[!] Database corrupted or unreadable. Starting fresh.")
        return {}

    def _save_db(self):
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.cache, f, indent=None) # Compact JSON
        except IOError as e:
            print(f"[!] Error saving hash database: {e}")

    def get_hashes(self, directory):
        """
        Scans directory, updates cache, and returns (paths, numpy_matrix).
        """
        print(f"[*] Scanning '{directory}'...")

        # 1. Scan filesystem
        current_files = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(full_path)
                        current_files[full_path] = mtime
                    except OSError:
                        continue

        if not current_files:
            return [], None

        # 2. Identify Changes
        paths_to_hash = []
        valid_paths = []

        # Check for modified or new files
        for path, mtime in current_files.items():
            valid_paths.append(path)
            if path not in self.cache:
                paths_to_hash.append(path)
            elif self.cache[path]['mtime'] != mtime:
                paths_to_hash.append(path) # File modified

        # 3. Clean up deleted files from DB
        # If it's in cache but not in current_files AND strictly inside this directory
        # (We must be careful not to delete entries from other directories if sharing DB)
        keys_to_remove = []
        abs_dir = os.path.abspath(directory)

        for cached_path in list(self.cache.keys()):
            # Only clean up files belonging to the directory we are currently scanning
            if os.path.abspath(cached_path).startswith(abs_dir):
                if cached_path not in current_files:
                    keys_to_remove.append(cached_path)

        for k in keys_to_remove:
            del self.cache[k]

        # 4. Process new/modified files
        if keys_to_remove:
            print(f"[*] Removed {len(keys_to_remove)} deleted files from cache.")

        if paths_to_hash:
            print(f"[*] Computing hashes for {len(paths_to_hash)} new/modified images...")
            with ProcessPoolExecutor() as executor:
                results = executor.map(_compute_single_hash, paths_to_hash)

                for i, (path, hash_str) in enumerate(results):
                    if hash_str:
                        self.cache[path] = {
                            'mtime': current_files[path],
                            'hash': hash_str
                        }
                    else:
                        # If hashing failed, ensure we don't keep bad data
                        if path in self.cache:
                            del self.cache[path]

                    if (i + 1) % 100 == 0:
                        print(f"    Hashing: {i + 1}/{len(paths_to_hash)}", end='\r')
            print("")
            self._save_db()
        else:
            print("[*] All hashes are up to date.")

        # 5. Build Result Matrix
        # We only return files that actually exist and successfully hashed
        final_paths = []
        hash_list = []

        for path in valid_paths:
            if path in self.cache and self.cache[path]['hash']:
                try:
                    # Convert Hex string back to Boolean Array for Numpy
                    # imagehash.hex_to_hash returns a ImageHash object, .hash is the bool array
                    h_obj = imagehash.hex_to_hash(self.cache[path]['hash'])
                    hash_list.append(h_obj.hash.flatten())
                    final_paths.append(path)
                except ValueError:
                    continue

        if not hash_list:
            return [], None

        return final_paths, np.array(hash_list)
