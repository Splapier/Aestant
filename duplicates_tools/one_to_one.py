#1:1 matching deleting
import os
import argparse
import time
import multiprocessing
import numpy as np
from duplicates_tools.hash_manager import HashManager

# --- Global shared memory for workers ---
shared_matrix = None

def init_worker(matrix):
    global shared_matrix
    shared_matrix = matrix

def worker_find_pairs(args):
    """
    Worker compares a specific range of rows against the REST of the matrix.
    To avoid double checking, row 'i' is only compared against rows > 'i'.
    """
    start_idx, end_idx, threshold = args
    local_pairs = []

    # We iterate over the assigned range
    for i in range(start_idx, end_idx):
        # Slice: compare current row 'i' ONLY with rows below it (i+1 onwards)
        # This prevents (A==B) and (B==A) duplication and self-matching.

        # Determine the slice of the matrix to compare against
        compare_slice = shared_matrix[i+1:]

        if compare_slice.shape[0] == 0:
            break

        dist = np.count_nonzero(compare_slice != shared_matrix[i], axis=1)
        matches = np.where(dist <= threshold)[0]

        for m_idx in matches:
            # The match index is relative to the slice.
            # Real index = (i + 1) + m_idx
            real_match_idx = (i + 1) + m_idx
            local_pairs.append((i, real_match_idx))

    return local_pairs

def find_duplicates_parallel(paths, matrix, threshold):
    print(f"[*] Searching for duplicates (Threshold: {threshold})...")
    start_time = time.time()

    num_images = len(paths)
    num_cores = multiprocessing.cpu_count()

    # Divide work: We split the outer loop range into chunks
    chunk_size = max(1, num_images // num_cores)
    ranges = []
    for i in range(0, num_images, chunk_size):
        ranges.append((i, min(i + chunk_size, num_images), threshold))

    print(f"[*] Dispatching to {num_cores} cores...")

    all_pairs = []
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(matrix,)) as pool:
        results = pool.map(worker_find_pairs, ranges)
        for res in results:
            all_pairs.extend(res)

    print(f"[*] Grouping {len(all_pairs)} matches...")

    # --- Grouping (Union-Find Logic) ---
    # Convert list of pairs [(0,5), (5,9)] into groups [(0,5,9)]
    adj = {}
    for x, y in all_pairs:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)

    visited = set()
    duplicate_groups = []

    for i in range(num_images):
        if i in visited:
            continue
        if i not in adj:
            continue

        # Found a new group, perform traversal (BFS)
        group_indices = []
        stack = [i]
        visited.add(i)
        while stack:
            curr = stack.pop()
            group_indices.append(curr)
            if curr in adj:
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        if len(group_indices) > 1:
            # Convert indices back to paths
            group_paths = [paths[idx] for idx in group_indices]
            duplicate_groups.append(group_paths)

    elapsed = time.time() - start_time
    print(f"[*] Comparison finished in {elapsed:.2f} seconds.")
    print(f"[*] Found {len(duplicate_groups)} sets of duplicate images.")
    return duplicate_groups

def delete_duplicates_smart(duplicate_groups):
    print("\n[*] Starting automatic deletion process...")
    deleted_count = 0
    bytes_saved = 0

    for group in duplicate_groups:
        files_data = []
        for path in group:
            try:
                size = os.path.getsize(path)
                files_data.append({'path': path, 'size': size})
            except OSError:
                continue

        if len(files_data) < 2: continue

        # Keep largest size, shortest path
        files_data.sort(key=lambda x: (-x['size'], len(x['path']), x['path']))
        keeper = files_data[0]
        trash = files_data[1:]

        print(f"    > Keeping: {os.path.basename(keeper['path'])}")
        for item in trash:
            try:
                os.remove(item['path'])
                print(f"      [DELETED] {item['path']}")
                deleted_count += 1
                bytes_saved += item['size']
            except OSError as e:
                print(f"      [ERROR] {e}")

    print(f"\n[*] Total Deleted: {deleted_count} | Space Saved: {bytes_saved / (1024*1024):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Strict Parallel Deduplicator")
    parser.add_argument("directory", type=str)
    parser.add_argument("-t", "--threshold", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print("[!] Directory not found.")
        return

    hm = HashManager()
    paths, matrix = hm.get_hashes(args.directory)

    if matrix is None: return

    duplicate_sets = find_duplicates_parallel(paths, matrix, args.threshold)
    if duplicate_sets:
        delete_duplicates_smart(duplicate_sets)
    else:
        print("\n--- No duplicates found. ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
