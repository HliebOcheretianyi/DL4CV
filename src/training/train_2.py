import os
from ultralytics import YOLO

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_yaml = os.path.join(base_path, "configs", "dataset.yaml")

    print("\n--- Фаза 1 вже завершена. Завантажуємо результати... ---")
    print("\n--- Запуск Фази 2: Unfreeze до 3 шару та Fine-tuning ---")

    path_to_best = os.path.join(base_path, "runs/detect/runs/train/phase1_frozen3/weights/best.pt")

    if not os.path.exists(path_to_best):
        print(f"ПОМИЛКА: Файл {path_to_best} не знайдено! Перевір шлях у провіднику.")
        return

    model_phase2 = YOLO(path_to_best)

    model_phase2.train(
        data=dataset_yaml,
        epochs=15,
        lr0=0.001,
        freeze=3,
        project="runs/train",
        name="phase2_finetune",
        device=0,
        workers=0,
        batch=8,
        exist_ok=True
    )

if __name__ == "__main__":
    main()