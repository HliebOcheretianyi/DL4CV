# verify_dataset.py — run this on every dataset before touching it
import os
from pathlib import Path
from collections import Counter
import cv2
import yaml


def audit_dataset(images_dir, labels_dir, yaml_path):
    images = set(Path(images_dir).glob("*.jpg")) | set(Path(images_dir).glob("*.png"))
    labels = set(Path(labels_dir).glob("*.txt"))

    # check for orphaned files (image without label or vice versa)
    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    missing_labels = img_stems - lbl_stems
    missing_images = lbl_stems - img_stems

    with open(yaml_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # Handle YAMLs where 'names' is a list instead of a dict
    if isinstance(data_loaded['names'], list):
        names = {i: name for i, name in enumerate(data_loaded['names'])}
    else:
        names = data_loaded['names']

    # class distribution
    class_counts = Counter()
    for lbl in labels:
        with open(lbl) as f:
            for line in f:
                parts = line.split()
                if parts:  # Added a quick check for empty lines
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    class_counts = Counter({names.get(old_key, old_key): values for old_key, values in class_counts.items()})

    print(f"\n--- Scanning: {images_dir} ---")
    print(f"Images: {len(images)}, Labels: {len(labels)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Class distribution: ")
    for label in list(class_counts.keys()):
        print(f"{label}: {class_counts[label]}")

    return missing_labels, missing_images, class_counts


if __name__ == "__main__":
    folders = ['../../data/raw/dataset-RAW', '../../data/raw/dataset-paulson']


    def fast_scandir(dirname):
        subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
        for dirname in list(subfolders):
            subfolders.extend(fast_scandir(dirname))
        return subfolders


    work_dirs = []
    for folder in folders:
        if os.path.exists(folder):
            # Use extend instead of append to keep a flat list of directories
            work_dirs.extend(fast_scandir(folder))

    # Loop through the directories you found
    for d in work_dirs:
        # Normalize slashes so we can easily find the 'images' folders
        d_norm = d.replace('\\', '/')

        # If we hit an images directory, derive the labels and yaml paths
        if d_norm.endswith('/images'):
            images_dir = d_norm
            labels_dir = d_norm.replace('/images', '/labels')

            # Go up two levels to find the data.yaml for this specific dataset
            parent_dataset_dir = os.path.dirname(os.path.dirname(images_dir))
            yaml_path = os.path.join(parent_dataset_dir, 'data.yaml')

            # Only audit if the labels folder actually exists (skips your missing test/labels issue)
            if os.path.exists(labels_dir) and os.path.exists(yaml_path):
                missing_labels, missing_images, class_counts = audit_dataset(images_dir, labels_dir, yaml_path)
            else:
                print(f"\n--- Skipping: {images_dir} (Missing labels folder or data.yaml) ---")