#----------------------find duplicates reference dir target dir
import os
import argparse
import multiprocessing
import time
import numpy as np
from duplicates_tools.hash_manager import HashManager

# --- Global var for worker processes to read memory without pickling overhead ---
shared_ref_matrix = None

def init_worker(ref_matrix):
    """Initialize worker with reference data."""
    global shared_ref_matrix
    shared_ref_matrix = ref_matrix

def worker_match_chunk(args):
    """
    Worker function to compare a chunk of target hashes against ALL reference hashes.
    """
    target_indices, target_sub_matrix, threshold = args
    results = []

    # Iterate through this chunk of targets
    for i, t_vec in enumerate(target_sub_matrix):
        # Compare this target vector against the global reference matrix
        distances = np.count_nonzero(shared_ref_matrix != t_vec, axis=1)

        # Find matches
        matches = np.where(distances <= threshold)[0]

        if matches.size > 0:
            # Find best match
            best_idx = matches[np.argmin(distances[matches])]
            dist = distances[best_idx]

            # Record the actual index in the main target list, ref index, and distance
            global_target_index = target_indices[i]
            results.append((global_target_index, best_idx, dist))

    return results

def find_and_delete_matches_parallel(ref_paths, ref_matrix, target_paths, target_matrix, threshold, dry_run=False):
    print(f"\n[*] Starting Parallel Matching (Threshold: {threshold})...")
    print(f"    References: {len(ref_paths)} | Targets: {len(target_paths)}")

    start_time = time.time()

    # Prepare chunks for multiprocessing
    num_cores = multiprocessing.cpu_count()
    total_targets = len(target_paths)

    # Split target matrix into chunks based on CPU count
    chunk_size = max(1, total_targets // num_cores)
    chunks = []

    for i in range(0, total_targets, chunk_size):
        end = min(i + chunk_size, total_targets)
        indices = range(i, end)
        sub_matrix = target_matrix[i:end]
        chunks.append((indices, sub_matrix, threshold))

    print(f"[*] Processing on {num_cores} cores...")

    files_to_delete = []

    # Run parallel pool
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(ref_matrix,)) as pool:
        # Map chunks to workers
        results_list = pool.map(worker_match_chunk, chunks)

        # Flatten results
        for chunk_res in results_list:
            for (t_idx, r_idx, dist) in chunk_res:
                files_to_delete.append((target_paths[t_idx], ref_paths[r_idx], dist))

    elapsed = time.time() - start_time
    print(f"\n[!] Found {len(files_to_delete)} matches in {elapsed:.2f} seconds.")
    print("-" * 60)

    # Deletion Phase
    for target_path, ref_path, dist in files_to_delete:
        print(f"Match (Dist: {dist}):")
        print(f"  Reference: {ref_path}")
        print(f"  Deleting:  {target_path}")

        if not dry_run:
            try:
                os.remove(target_path)
                print("  [DELETED]")
            except Exception as e:
                print(f"  [ERROR] Could not delete file: {e}")
        else:
            print("  [DRY RUN - File not deleted]")
        print("-" * 20)

def main():
    parser = argparse.ArgumentParser(description="Fast Image Deduplicator (Parallel Matching)")
    parser.add_argument("reference_dir", type=str)
    parser.add_argument("target_dir", type=str)
    parser.add_argument("-t", "--threshold", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(args.reference_dir) or not os.path.isdir(args.target_dir):
        print("[!] Error: Directories do not exist.")
        return

    hm = HashManager()

    # Get Data
    ref_paths, ref_matrix = hm.get_hashes(args.reference_dir)
    target_paths, target_matrix = hm.get_hashes(args.target_dir)

    if ref_matrix is None or target_matrix is None:
        print("[!] Missing images.")
        return

    find_and_delete_matches_parallel(ref_paths, ref_matrix, target_paths, target_matrix, args.threshold, args.dry_run)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
