import random
import shutil
from collections import Counter
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────
SEED = 42
LABELS_DIR = Path("../../data/final/labels")
IMAGES_DIR = Path("../../data/final/images")

CAP_CONFIG = {
    6: 20000,   # civilian_vehicle — was 211,468, cap to match military classes
}

CLASSES = {
    0: "tank",
    1: "apc_ifv",
    2: "armored_car",
    3: "artillery",
    4: "logistics_truck",
    5: "soldier",
    6: "civilian_vehicle",
}

# ─── HELPERS ──────────────────────────────────────────────
def count_classes(labels_dir):
    counts = Counter()
    for lbl in Path(labels_dir).glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[int(line.split()[0])] += 1
    return counts


def print_distribution(title, counts):
    print(f"\n  {title}")
    total = sum(counts.values())
    for class_id, class_name in CLASSES.items():
        n = counts.get(class_id, 0)
        pct = (n / total * 100) if total > 0 else 0
        bar = "█" * (n // 500)
        print(f"    [{class_id}] {class_name:<20} {n:>8}  ({pct:5.1f}%)  {bar}")
    print(f"    {'TOTAL':<22} {total:>8}")


def backup_labels(labels_dir, backup_dir):
    backup_dir = Path(backup_dir)
    if backup_dir.exists():
        print(f"\n[INFO] Backup already exists at {backup_dir}, skipping.")
        return
    shutil.copytree(labels_dir, backup_dir)
    print(f"\n[INFO] Backup created at {backup_dir}")


def cap_class(labels_dir, class_id, max_annotations, seed=42):
    random.seed(seed)
    labels_dir = Path(labels_dir)
    label_files = list(labels_dir.glob("*.txt"))

    file_contents = {}
    target_indices = []

    for lbl in label_files:
        with open(lbl) as f:
            lines = [l.strip() for l in f.readlines()]
        file_contents[lbl] = lines
        for i, line in enumerate(lines):
            if line and int(line.split()[0]) == class_id:
                target_indices.append((lbl, i))

    current_count = len(target_indices)
    class_name = CLASSES.get(class_id, str(class_id))

    print(f"\n[CAP] Class {class_id} ({class_name})")
    print(f"  Before: {current_count} annotations")
    print(f"  Target: {max_annotations} annotations")

    if current_count <= max_annotations:
        print(f"  Already under cap — nothing to do.")
        return

    n_to_drop = current_count - max_annotations
    to_drop = set()
    drop_sample = random.sample(target_indices, n_to_drop)
    for file_path, line_idx in drop_sample:
        to_drop.add((file_path, line_idx))

    print(f"  Dropping: {n_to_drop} annotations across {len(label_files)} files")

    files_modified = 0
    files_emptied = 0

    for lbl, lines in file_contents.items():
        new_lines = []
        for i, line in enumerate(lines):
            if (lbl, i) in to_drop:
                continue
            if line:
                new_lines.append(line)

        with open(lbl, "w") as f:
            f.write("\n".join(new_lines))

        if len(new_lines) < len([l for l in lines if l]):
            files_modified += 1
        if len(new_lines) == 0:
            files_emptied += 1

    print(f"  Files modified: {files_modified}")
    if files_emptied > 0:
        print(f"  [WARN] {files_emptied} label files are now empty")

    return files_emptied


def remove_empty_labels(labels_dir, images_dir):
    """Remove empty label files and their corresponding images."""
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    removed = 0

    for lbl in labels_dir.glob("*.txt"):
        with open(lbl) as f:
            content = f.read().strip()
        if not content:
            lbl.unlink()
            for ext in [".jpg", ".jpeg", ".png"]:
                img = images_dir / (lbl.stem + ext)
                if img.exists():
                    img.unlink()
                    break
            removed += 1

    print(f"\n[CLEANUP] Removed {removed} empty label files and their images")
    return removed


# ─── MAIN ─────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  CLASS CAPPING — TRAIN SPLIT ONLY")
    print("="*55)
    print("  Val and test splits are never modified.")

    train_labels = LABELS_DIR / "train"
    train_images = IMAGES_DIR / "train"

    # 1. Backup
    backup_labels(train_labels, LABELS_DIR / "train_backup")

    # 2. Before distribution
    before = count_classes(train_labels)
    print_distribution("BEFORE CAPPING", before)

    # 3. Apply caps
    total_emptied = 0
    for class_id, max_annotations in CAP_CONFIG.items():
        emptied = cap_class(train_labels, class_id, max_annotations, seed=SEED)
        if emptied:
            total_emptied += emptied

    # 4. Remove empty label files and their images
    if total_emptied > 0:
        remove_empty_labels(train_labels, train_images)

    # 5. After distribution
    after = count_classes(train_labels)
    print_distribution("AFTER CAPPING", after)

    # 6. Confirm val and test untouched
    print("\n  Val and test splits (should be unchanged):")
    for split in ["val", "test"]:
        counts = count_classes(LABELS_DIR / split)
        total = sum(counts.values())
        print(f"    {split}: {total} total annotations — untouched ✅")

    # 7. Final summary
    print("\n[DONE] Capping complete.")
    for class_id, max_annotations in CAP_CONFIG.items():
        class_name = CLASSES.get(class_id, str(class_id))
        print(f"  {class_name}: {before.get(class_id,0):,} -> {after.get(class_id,0):,}")

    # 8. Final image/label count
    remaining_images = len(list(train_images.glob("*.jpg"))) + \
                       len(list(train_images.glob("*.png")))
    remaining_labels = len(list(train_labels.glob("*.txt")))
    print(f"\n  Remaining train images: {remaining_images}")
    print(f"  Remaining train labels: {remaining_labels}")

    if remaining_images != remaining_labels:
        print(f"  [WARN] Image/label count mismatch — investigate!")
    else:
        print(f"  Image/label counts match ✅")

    print("="*55 + "\n")


if __name__ == "__main__":
    main()