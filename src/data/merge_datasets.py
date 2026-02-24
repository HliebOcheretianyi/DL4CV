import random
import shutil
from collections import Counter
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────
SEED = 42
SPLIT = (0.80, 0.10, 0.10)  # train / val / test
OUTPUT_DIR = Path("data/final")

SOURCES = [
    {
        "name": "Paulson",
        "splits": {
            "train": ("../../data/processed/dataset-paulson/train/images",
                      "../../data/processed/dataset-paulson/train/labels"),
            "valid": ("../../data/processed/dataset-paulson/valid/images",
                      "../../data/processed/dataset-paulson/valid/labels"),
        }
    },
    {
        "name": "RAW",
        "splits": {
            "train": ("../../data/processed/dataset-RAW/train/images",
                      "../../data/processed/dataset-RAW/train/labels"),
            "val": ("../../data/processed/dataset-RAW/val/images",
                    "../../data/processed/dataset-RAW/val/labels"),
            "test": ("../../data/processed/dataset-RAW/test/images",
                     "../../data/processed/dataset-RAW/test/labels"),
        }
    },
]

UNIFIED_CLASSES = {
    0: "tank",
    1: "apc_ifv",
    2: "armored_car",
    3: "artillery",
    4: "logistics_truck",
    5: "soldier",
    6: "civilian_vehicle",
}


# ─── HELPERS ──────────────────────────────────────────────
def get_pairs(images_dir, labels_dir):
    """Get matched image-label pairs, skip if either is missing."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    pairs = []
    for img in images_dir.glob("*"):
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            print(f"  [SKIP] No label for {img.name}")
            continue
        if lbl.stat().st_size == 0:
            print(f"  [SKIP] Empty label for {img.name}")
            continue
        pairs.append((img, lbl))
    return pairs


def copy_pair(img_path, lbl_path, dest_images, dest_labels, new_stem):
    """Copy image and label to destination with a unique stem."""
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    img_ext = img_path.suffix.lower()
    shutil.copy(img_path, dest_images / f"{new_stem}{img_ext}")
    shutil.copy(lbl_path, dest_labels / f"{new_stem}.txt")


def split_pairs(pairs, split_ratios, seed):
    """Shuffle and split pairs into train/val/test."""
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    return (
        pairs[:n_train],
        pairs[n_train:n_train + n_val],
        pairs[n_train + n_val:]
    )


def count_classes(labels_dir):
    counts = Counter()
    for lbl in Path(labels_dir).glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[int(line.split()[0])] += 1
    return counts


def print_report(title, counts, total_images):
    print(f"\n  {title}")
    print(f"  Images: {total_images}")
    for class_id, class_name in UNIFIED_CLASSES.items():
        n = counts.get(class_id, 0)
        bar = "█" * (n // 100)
        print(f"    [{class_id}] {class_name:<20} {n:>6}  {bar}")


# ─── MAIN ─────────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  MILITARY DETECTION — DATASET MERGE")
    print("=" * 55)

    # 1. Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"\n[INFO] Cleared existing {OUTPUT_DIR}")

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Collect all pairs per source and re-split
    total_copied = {"train": 0, "val": 0, "test": 0}

    for source in SOURCES:
        name = source["name"]
        print(f"\n[SOURCE] {name}")

        # Pool all pairs from all splits of this source
        all_pairs = []
        for split_name, (img_dir, lbl_dir) in source["splits"].items():
            pairs = get_pairs(img_dir, lbl_dir)
            print(f"  {split_name}: {len(pairs)} valid pairs")
            all_pairs.extend(pairs)

        print(f"  Total pooled: {len(all_pairs)} pairs")

        # Re-split by source to prevent leakage
        train_pairs, val_pairs, test_pairs = split_pairs(
            all_pairs, SPLIT, seed=SEED
        )

        # 3. Copy with unique filenames (source prefix to avoid collisions)
        for pairs, split_name in [
            (train_pairs, "train"),
            (val_pairs, "val"),
            (test_pairs, "test"),
        ]:
            for i, (img, lbl) in enumerate(pairs):
                new_stem = f"{name.lower()}_{split_name}_{i:06d}"
                copy_pair(
                    img, lbl,
                    OUTPUT_DIR / "images" / split_name,
                    OUTPUT_DIR / "labels" / split_name,
                    new_stem
                )
            total_copied[split_name] += len(pairs)
            print(f"  -> {split_name}: {len(pairs)} copied")

    # 4. Final audit
    print("\n" + "=" * 55)
    print("  FINAL DATASET REPORT")
    print("=" * 55)

    for split in ["train", "val", "test"]:
        images = list((OUTPUT_DIR / "images" / split).glob("*"))
        labels = list((OUTPUT_DIR / "labels" / split).glob("*.txt"))
        counts = count_classes(OUTPUT_DIR / "labels" / split)

        # sanity check
        if len(images) != len(labels):
            print(f"\n  [WARNING] {split}: image/label count mismatch!")
            print(f"  Images: {len(images)} | Labels: {len(labels)}")

        print_report(f"SPLIT: {split.upper()}", counts, len(images))

    # 5. Write dataset.yaml
    yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(UNIFIED_CLASSES)}
names:
"""
    for class_id, class_name in UNIFIED_CLASSES.items():
        yaml_content += f"  {class_id}: {class_name}\n"

    yaml_path = Path("configs/dataset.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[INFO] dataset.yaml written to {yaml_path}")
    print("\n[DONE] Merge complete.")
    print(f"  Train: {total_copied['train']} | Val: {total_copied['val']} | Test: {total_copied['test']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()