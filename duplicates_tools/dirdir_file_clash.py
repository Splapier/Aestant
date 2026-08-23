import os

def rename_conflicts(dir_a, dir_b):
    """
    Renames files in dir_a that have conflicting filenames in dir_b.
    """

    # 1. Validate directories exist
    if not os.path.exists(dir_a) or not os.path.exists(dir_b):
        print("Error: One or both directories do not exist.")
        return

    print(f"Checking for conflicts between:\n A: {dir_a}\n B: {dir_b}\n")

    # 2. Get list of files in Directory A
    files_in_a = os.listdir(dir_a)

    renamed_count = 0

    for filename in files_in_a:
        # Construct full file paths
        path_a = os.path.join(dir_a, filename)
        path_b = os.path.join(dir_b, filename)

        # Check if it is a file (ignore subdirectories)
        if os.path.isfile(path_a):

            # 3. Check if the file also exists in Directory B
            if os.path.exists(path_b):

                # Split name and extension (e.g., 'picture' and '.jpg')
                name, ext = os.path.splitext(filename)

                counter = 1
                new_filename = f"{name}_renamed_{counter}{ext}"
                new_path_a = os.path.join(dir_a, new_filename)

                # 4. Find a unique name
                # We must ensure the new name doesn't exist in B (still a conflict)
                # AND doesn't exist in A (don't overwrite another local file)
                while os.path.exists(new_path_a) or os.path.exists(os.path.join(dir_b, new_filename)):
                    counter += 1
                    new_filename = f"{name}_renamed_{counter}{ext}"
                    new_path_a = os.path.join(dir_a, new_filename)

                # 5. Rename the file
                try:
                    os.rename(path_a, new_path_a)
                    print(f"[Renamed]: {filename} -> {new_filename}")
                    renamed_count += 1
                except OSError as e:
                    print(f"[Error] Could not rename {filename}: {e}")

    if renamed_count == 0:
        print("\nNo conflicts found.")
    else:
        print(f"\nProcess complete. {renamed_count} files renamed.")

# --- Configuration ---
# You can change these paths to the actual folders you want to use
directory_a = "./dataset/all_images"
directory_b = "./dataset/images"

# --- Execution ---
if __name__ == "__main__":
    # Create dummy folders for testing purposes if they don't exist
    # (You can remove this block when using real folders)
    if not os.path.exists(directory_a):
        print("Note: Please set the 'directory_a' and 'directory_b' variables to real paths.")
    else:
        rename_conflicts(directory_a, directory_b)
