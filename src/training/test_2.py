from ultralytics import YOLO
import os


def validate():
    model_path = r"F:\3 course\dl\runs\detect\runs\train\phase2_finetune\weights\best.pt"
    dataset_yaml = r"F:\3 course\dl\configs\dataset.yaml"

    if not os.path.exists(model_path):
        print("Модель не знайдена")
        return

    model = YOLO(model_path)

    print("\n--- ЗАПУСК ПОРІВНЯННЯ ТЕСТОВИХ ДАНИХ З ЛЕЙБЛАМИ ---")

    metrics = model.val(
        data=dataset_yaml,
        split='test',
        project="runs/val",
        name="final_test_report",
        device=0,
        verbose=True
    )

    print("\n" + "=" * 50)
    print(f"РЕЗУЛЬТАТИ ТЕСТУВАННЯ (mAP@0.5): {metrics.box.map50:.4f}")
    print("=" * 50)

    for i, name in enumerate(metrics.names.values()):
        cls_map = metrics.box.class_result(i)[2]
        print(f"Клас: {name:15} | mAP@0.5: {cls_map:.4f}")

    print("=" * 50)
    print(f"Результати: {metrics.save_dir}")


if __name__ == "__main__":
    validate()
