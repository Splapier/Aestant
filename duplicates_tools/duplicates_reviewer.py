# review_duplicates.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import shutil

# --- Configuration ---
THUMBNAIL_SIZE = (400, 400)
DELETED_FOLDER = "deleted_duplicates" # A folder to move duplicates to
WHITELIST_FILE = "whitelist_duplicates.json"     # File to store ignored groups

class DuplicateReviewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Duplicate Image Reviewer")

        self.duplicate_groups = []
        self.whitelisted_groups = []
        self.current_group_index = -1
        self.keeper_var = tk.StringVar() # Holds the path of the image to KEEP

        # Load existing whitelist if it exists
        self.load_whitelist()

        # --- GUI Setup ---
        # Top frame for file loading
        top_frame = tk.Frame(root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        load_button = tk.Button(top_frame, text="Load duplicates.json File", command=self.load_file)
        load_button.pack(side=tk.LEFT)

        # Info label about whitelist
        self.stats_label = tk.Label(top_frame, text=f"Whitelist entries: {len(self.whitelisted_groups)}")
        self.stats_label.pack(side=tk.RIGHT)

        # Main frame for displaying images
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.group_label = tk.Label(self.main_frame, text="Load a file to begin.", font=("Helvetica", 14))
        self.group_label.pack(pady=20)

        # Bottom frame for navigation and actions
        bottom_frame = tk.Frame(root, padx=10, pady=10)
        bottom_frame.pack(fill=tk.X)

        # Navigation
        nav_frame = tk.Frame(bottom_frame)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.prev_button = tk.Button(nav_frame, text="<< Previous Group", command=self.prev_group, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(nav_frame, text="Next Group >>", command=self.next_group, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT)

        # Actions
        action_frame = tk.Frame(bottom_frame)
        action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.whitelist_button = tk.Button(action_frame, text="Whitelist Group (Ignore)", command=self.whitelist_current_group, state=tk.DISABLED, bg="#ffffe0")
        self.whitelist_button.pack(side=tk.LEFT, padx=20)

        self.process_button = tk.Button(action_frame, text="Process Group (Keep Selected, Delete Rest)", command=self.process_selection, state=tk.DISABLED, bg="#e0ffe0")
        self.process_button.pack(side=tk.RIGHT, padx=20)

    def load_whitelist(self):
        if os.path.exists(WHITELIST_FILE):
            try:
                with open(WHITELIST_FILE, 'r') as f:
                    self.whitelisted_groups = json.load(f)
                # Ensure they are sorted for consistent comparison later
                self.whitelisted_groups = [sorted(g) for g in self.whitelisted_groups]
            except Exception:
                self.whitelisted_groups = []

    def save_whitelist(self):
        try:
            with open(WHITELIST_FILE, 'w') as f:
                json.dump(self.whitelisted_groups, f, indent=4)
            self.stats_label.config(text=f"Whitelist entries: {len(self.whitelisted_groups)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save whitelist:\n{e}")

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select duplicates.json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                raw_groups = json.load(f)

            if not raw_groups:
                messagebox.showinfo("Info", "The selected file contains no duplicate groups.")
                return

            # Filter out whitelisted groups
            self.duplicate_groups = []

            # Convert whitelist to a set of frozensets for faster lookup
            whitelist_set = set(frozenset(g) for g in self.whitelisted_groups)

            ignored_count = 0
            for group in raw_groups:
                if frozenset(group) in whitelist_set:
                    ignored_count += 1
                else:
                    self.duplicate_groups.append(group)

            if ignored_count > 0:
                print(f"Ignored {ignored_count} groups found in whitelist.")

            if not self.duplicate_groups:
                messagebox.showinfo("Info", "All groups in this file are whitelisted!")
                self.group_label.config(text="All loaded groups are whitelisted.")
                return

            self.current_group_index = 0
            self.display_current_group()
            self.update_button_states()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or parse the file:\n{e}")

    def display_current_group(self):
        # Clear previous images
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        if not self.duplicate_groups or self.current_group_index < 0:
            self.group_label = tk.Label(self.main_frame, text="No more groups to review.", font=("Helvetica", 14))
            self.group_label.pack(pady=20)
            return

        # Update label
        self.group_label = tk.Label(self.main_frame,
                                    text=f"Group {self.current_group_index + 1} of {len(self.duplicate_groups)}",
                                    font=("Helvetica", 14))
        self.group_label.pack(pady=10)

        # Create a frame for the images with a canvas and scrollbar
        canvas = tk.Canvas(self.main_frame)
        scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Display images in the current group
        group = self.duplicate_groups[self.current_group_index]

        # Default the selection to the first image if not already set or invalid
        if not self.keeper_var.get() in group:
            self.keeper_var.set(group[0])

        for img_path in group:
            container = tk.Frame(scrollable_frame, borderwidth=2, relief="groove")
            container.pack(pady=5, padx=5, fill=tk.X)

            # Left side: Controls
            controls_frame = tk.Frame(container)
            controls_frame.pack(side=tk.LEFT, padx=10)

            # 1. Radio Button for "Keep This"
            rb = tk.Radiobutton(controls_frame, text="Keep This", variable=self.keeper_var, value=img_path, font=("Arial", 10, "bold"))
            rb.pack(anchor="w", pady=5)

            # 2. Individual Delete Button
            # We use lambda p=img_path: to capture the specific path for this iteration
            del_btn = tk.Button(controls_frame, text="Delete File", bg="#ffcccc", fg="red",
                                command=lambda p=img_path: self.delete_single_image(p))
            del_btn.pack(anchor="w", pady=5)

            try:
                img = Image.open(img_path)
                img.thumbnail(THUMBNAIL_SIZE)
                photo = ImageTk.PhotoImage(img)

                img_label = tk.Label(container, image=photo)
                img_label.image = photo # Keep a reference!
                img_label.pack(side=tk.LEFT)

                # File info
                file_size_mb = os.path.getsize(img_path) / (1024 * 1024)
                info_text = f"Path: {img_path}\nDims: {img.size[0]}x{img.size[1]}px\nSize: {file_size_mb:.2f} MB"
                info_label = tk.Label(container, text=info_text, justify=tk.LEFT)
                info_label.pack(side=tk.LEFT, padx=10)

            except Exception as e:
                error_label = tk.Label(container, text=f"Could not load:\n{os.path.basename(img_path)}\n{e}", fg="red")
                error_label.pack(side=tk.LEFT, padx=10)

    def next_group(self):
        if self.current_group_index < len(self.duplicate_groups) - 1:
            self.current_group_index += 1
            self.display_current_group()
            self.update_button_states()

    def prev_group(self):
        if self.current_group_index > 0:
            self.current_group_index -= 1
            self.display_current_group()
            self.update_button_states()

    def update_button_states(self):
        has_groups = len(self.duplicate_groups) > 0
        self.prev_button['state'] = tk.NORMAL if self.current_group_index > 0 else tk.DISABLED
        self.next_button['state'] = tk.NORMAL if self.current_group_index < len(self.duplicate_groups) - 1 else tk.DISABLED
        self.process_button['state'] = tk.NORMAL if has_groups else tk.DISABLED
        self.whitelist_button['state'] = tk.NORMAL if has_groups else tk.DISABLED

    def whitelist_current_group(self):
        """Adds the current group to the whitelist and removes it from view."""
        if self.current_group_index < 0 or not self.duplicate_groups:
            return

        current_group = self.duplicate_groups[self.current_group_index]

        # Add to whitelist (sorted to ensure consistency)
        self.whitelisted_groups.append(sorted(current_group))
        self.save_whitelist()

        # Remove from current view
        self.remove_current_group_from_view()

    def move_file_to_trash(self, path):
        """Helper to safely move file to deleted folder with rename on collision."""
        if not os.path.exists(DELETED_FOLDER):
            os.makedirs(DELETED_FOLDER)

        filename = os.path.basename(path)
        dest_path = os.path.join(DELETED_FOLDER, filename)

        # If file exists in delete folder, append a counter
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(DELETED_FOLDER, f"{base}_{counter}{ext}")
            counter += 1

        shutil.move(path, dest_path)
        return True

    def delete_single_image(self, path):
        """Deletes a single image from the current group immediately."""
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete:\n{os.path.basename(path)}?"):
            try:
                self.move_file_to_trash(path)

                # Update internal data
                current_group = self.duplicate_groups[self.current_group_index]
                if path in current_group:
                    current_group.remove(path)

                # Check if group is still valid (needs at least 2 to be a 'duplicate group')
                if len(current_group) < 2:
                    # If less than 2 left, the group is resolved (or invalid). Move to next.
                    self.duplicate_groups.pop(self.current_group_index)

                    # Fix index bounds
                    if self.current_group_index >= len(self.duplicate_groups):
                        self.current_group_index = len(self.duplicate_groups) - 1
                    if not self.duplicate_groups:
                        self.current_group_index = -1

                self.update_button_states()
                self.display_current_group()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file:\n{e}")

    def process_selection(self):
        """Moves all images NOT selected by the radio button to the delete folder."""
        if self.current_group_index < 0 or not self.duplicate_groups:
            return

        keeper_path = self.keeper_var.get()
        current_group = self.duplicate_groups[self.current_group_index]

        if keeper_path not in current_group:
            messagebox.showerror("Error", "Selected image is not part of the current group.")
            return

        # Identify items to delete
        images_to_delete = [p for p in current_group if p != keeper_path]

        moved_count = 0
        for path in images_to_delete:
            try:
                self.move_file_to_trash(path)
                moved_count += 1
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not move file {path}:\n{e}")

        self.remove_current_group_from_view()

    def remove_current_group_from_view(self):
        """Helper to remove data and refresh GUI"""
        self.duplicate_groups.pop(self.current_group_index)

        # Adjust index if we removed the last item
        if self.current_group_index >= len(self.duplicate_groups):
            self.current_group_index = len(self.duplicate_groups) - 1

        if not self.duplicate_groups:
            self.current_group_index = -1

        self.update_button_states()
        self.display_current_group()


if __name__ == "__main__":
    root = tk.Tk()
    app = DuplicateReviewerApp(root)
    root.geometry("900x700")
    root.mainloop()
