from ultralytics import YOLO
import os

model_path = r"F:\3 course\dl\runs\detect\runs\train\phase2_finetune\weights\best.pt"
model = YOLO(model_path)

source = r"F:\3 course\dl\data\final\images\test"

results = model.predict(
    source=source,
    conf=0.25,
    save=True,
    project="runs/detect",
    name="my_tests"
)

print(f"Готово! Результати збережено в папку: runs/detect/my_tests")