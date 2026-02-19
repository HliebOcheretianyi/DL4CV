from pathlib import Path


PAULSON_REMAP = {
    0: 2,    # ARMOREED_CAR -> armored_car
    1: 6,    # CIVILIAN_CAR -> civilian_vehicle
    2: 1,    # ICV -> apc_ifv
    3: 5,    # PERSON -> soldier
    4: 0,    # TANK -> tank
    5: 4,    # TRUCK -> logistics_truck
}

RAW_REMAP = {
    0: 5,    # camouflage_soldier -> soldier
    1: None, # weapon -> drop
    2: 0,    # military_tank -> tank
    3: 4,    # military_truck -> logistics_truck
    4: 1,    # military_vehicle -> apc_ifv
    5: 5,    # civilian -> soldier
    6: 5,    # soldier -> soldier
    7: 6,    # civilian_vehicle -> civilian_vehicle
    8: 3,    # military_artillery -> artillery
    9: None, # trench -> drop
    10: None, # military_aircraft -> drop
    11: None, # military_warship -> drop
}

THREAT_LEVELS = {
    "tank": "HIGH",                     # 0
    "apc_ifv": "HIGH",                  # 1
    "armored_car": "HIGH",              # 2
    "artillery": "MEDIUM",              # 3
    "logistics_truck": "MEDIUM",        # 4
    "soldier": "MEDIUM",                # 5
    "civilian_vehicle": "NONE"          # 6          no alert triggered
}

def remap_label_file(src_path, dst_path, remap_dict):
    with open(src_path) as f:
        lines = f.readlines()

    remapped = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        original_class = int(parts[0])
        new_class = remap_dict.get(original_class, None)

        if new_class is None:
            continue  # drop this annotation

        parts[0] = str(new_class)
        remapped.append(" ".join(parts))

    with open(dst_path, "w") as f:
        f.write("\n".join(remapped))


def remap_dataset(labels_dir, output_dir, remap_dict):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for label_file in Path(labels_dir).glob("*.txt"):
        remap_label_file(
            label_file,
            Path(output_dir) / label_file.name,
            remap_dict
        )


# run it
remap_dataset("../../data/raw/dataset-paulson/train/labels",
              "../../data/processed/dataset-paulson/train/labels",
              PAULSON_REMAP)

remap_dataset("../../data/raw/dataset-RAW/train/labels",
              "../../data/processed/dataset-RAW/train/labels",
              RAW_REMAP)