import os
from ultralytics import YOLO
import torch


# from utils.checks import check_det_dataset

def main():
    # Отримуємо абсолютний шлях до поточної папки проєкту
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_yaml = os.path.join(base_path, "configs", "dataset.yaml")

    print(f"Шукаю конфіг за шляхом: {dataset_yaml}")

    if not os.path.exists(dataset_yaml):
        print(f"Помилка: Файл не знайдено за шляхом {dataset_yaml}!")
        return

    # фаза 1
    print("\n--- Запуск Фази 1: Заморожуємо Backbone (10 шарів) ---")
    model = YOLO("yolov8n.pt")

    model.train(
        data=dataset_yaml,
        epochs=40,
        lr0=0.01,
        freeze=10,
        project="runs/train",
        name="phase1_frozen",
        device=0,
        workers=0,
        batch=8,
        imgsz=640
    )

    # фаза 2
    print("\n--- Запуск Фази 2: Unfreeze до 3 шару та Fine-tuning ---")
    model_phase2 = YOLO("runs/train/phase1_frozen/weights/best.pt")

    model_phase2.train(
        data=dataset_yaml,
        epochs=15,
        lr0=0.001,
        freeze=3,
        project="runs/train",
        name="phase2_finetune",
        device=0,
        workers=0,
        batch=8
    )

if __name__ == "__main__":
    main()
