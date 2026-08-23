import os
import json
import argparse
import time
import multiprocessing
import numpy as np
from duplicates_tools.hash_manager import HashManager

# --- Global Shared ---
shared_matrix = None

def init_worker(matrix):
    global shared_matrix
    shared_matrix = matrix

def worker_find_pairs(args):
    start_idx, end_idx, threshold = args
    local_pairs = []

    for i in range(start_idx, end_idx):
        compare_slice = shared_matrix[i+1:]
        if compare_slice.shape[0] == 0: break

        dist = np.count_nonzero(compare_slice != shared_matrix[i], axis=1)
        matches = np.where(dist <= threshold)[0]

        for m_idx in matches:
            real_match_idx = (i + 1) + m_idx
            local_pairs.append((i, real_match_idx))
    return local_pairs

def find_duplicates_parallel(paths, matrix, threshold):
    print(f"[*] Searching for duplicates (Threshold: {threshold})...")
    start_time = time.time()

    num_images = len(paths)
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, num_images // num_cores)
    ranges = [(i, min(i + chunk_size, num_images), threshold) for i in range(0, num_images, chunk_size)]

    print(f"[*] Dispatching to {num_cores} cores...")

    all_pairs = []
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(matrix,)) as pool:
        results = pool.map(worker_find_pairs, ranges)
        for res in results:
            all_pairs.extend(res)

    print(f"[*] Grouping {len(all_pairs)} matches...")

    # Graph traversal to group
    adj = {}
    for x, y in all_pairs:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)

    visited = set()
    duplicate_groups = []

    for i in range(num_images):
        if i in visited or i not in adj: continue

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
            duplicate_groups.append([paths[idx] for idx in group_indices])

    elapsed = time.time() - start_time
    print(f"[*] Comparison finished in {elapsed:.2f} seconds.")
    print(f"[*] Found {len(duplicate_groups)} sets of duplicate images.")
    return duplicate_groups

def main():
    parser = argparse.ArgumentParser(description="Parallel Duplicate Finder (JSON)")
    parser.add_argument("directory", type=str)
    parser.add_argument("-t", "--threshold", type=int, default=5)
    parser.add_argument("-o", "--output", type=str, default="duplicates.json")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print("[!] Directory not found.")
        return

    hm = HashManager()
    paths, matrix = hm.get_hashes(args.directory)
    if matrix is None: return

    duplicate_sets = find_duplicates_parallel(paths, matrix, args.threshold)

    if duplicate_sets:
        try:
            with open(args.output, 'w') as f:
                json.dump(duplicate_sets, f, indent=4)
            print(f"[+] Saved to '{args.output}'")
        except Exception as e:
            print(f"[!] Error saving JSON: {e}")
    else:
        print("\n--- No duplicates found. ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

