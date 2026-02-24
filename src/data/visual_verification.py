import random
from pathlib import Path

import cv2

CLASSES = {
    0: "tank", 1: "apc_ifv", 2: "armored_car",
    3: "artillery", 4: "logistics_truck",
    5: "soldier", 6: "civilian_vehicle"
}
COLORS = {
    0: (0,0,255), 1: (0,0,200), 2: (0,0,180),
    3: (0,165,255), 4: (0,200,255),
    5: (0,255,255), 6: (200,200,200)
}

def verify(images_dir, labels_dir, n=30):
    images = list(Path(images_dir).glob("*.jpg")) + \
             list(Path(images_dir).glob("*.png"))
    sample = random.sample(images, min(n, len(images)))

    for img_path in sample:
        lbl_path = Path(labels_dir) / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = COLORS.get(cls, (255,255,255))
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(img, CLASSES.get(cls, f"cls{cls}"),
                               (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, color, 2)

        cv2.imshow("Verify any key: next | Q: quit", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

# Paulson
print("Checking...")
verify(
    "../../data/final/images/test",
    "../../data/final/labels/test",
)
