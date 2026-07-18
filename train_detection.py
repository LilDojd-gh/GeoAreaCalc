import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Импортируем компоненты из нашего пакета src
from src import VehicleDetectionDataset, get_vehicle_detection_model
from src.dataset import detection_transform  # Импортируем аугментации для детекции


def collate_fn(batch):
    """
    Специальная функция для DataLoader.
    Поскольку на разных снимках находится разное количество машин,
    стандартный сборщик PyTorch не сможет объединить их в фиксированную матрицу.
    Эта функция упаковывает батч в кортеж (tuple).
    """
    return tuple(zip(*batch))


def train_detection():
    # ------------------------------------------
    # 1. Настройки гиперпараметров
    # ------------------------------------------
    EPOCHS = 15
    BATCH_SIZE = 4  # Модель детекции тяжелая, берем батч поменьше для экономии памяти
    LEARNING_RATE = 5e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Запуск обучения Faster R-CNN на устройстве: {DEVICE}")

    # ------------------------------------------
    # 2. Подготовка пайплайна данных
    # ------------------------------------------
    # Пути к изображениям и текстовым файлам разметки YOLO
    train_image_dir = "data/vehicles_dataset/images/"
    train_label_dir = "data/vehicles_dataset/labels/"

    train_dataset = VehicleDetectionDataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        transform=detection_transform
    )

    # Инициализируем DataLoader ОБЯЗАТЕЛЬНО с нашей collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == "cuda" else False
    )

    # ------------------------------------------
    # 3. Инициализация компонентов модели
    # ------------------------------------------
    # 2 класса: 0 - фоновый класс, 1 - автомобиль
    model = get_vehicle_detection_model(num_classes=2).to(DEVICE)

    # Оптимизатор AdamW с затуханием весов (weight decay) для предотвращения переобучения
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    os.makedirs("weights", exist_ok=True)
    best_loss = float('inf')

    # ------------------------------------------
    # 4. Главный цикл обучения (Training Loop)
    # ------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()  # Переводим модель в режим обучения
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for images, targets in loop:
            # Переносим список картинок на устройство
            images = list(image.to(DEVICE) for image in images)
            # Переносим словари с рамками и метками на устройство
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # --- Forward Pass ---
            # В режиме train() Faster R-CNN вместо предсказаний выдает словарь с лоссами:
            # - loss_classifier (ошибка классификации объекта)
            # - loss_box_reg (ошибка точности координат рамки)
            # - loss_objectness и loss_rpn_box_reg (ошибки подсети предложений регионов RPN)
            loss_dict = model(images, targets)

            # Суммируем все эти лоссы (комбинация лоссов по ТЗ)
            losses = sum(loss for loss in loss_dict.values())

            # --- Backward Pass ---
            optimizer.zero_grad()  # Сброс старых градиентов
            losses.backward()  # Расчет новых градиентов по сумме всех лоссов
            optimizer.step()  # Шаг оптимизатора (обновление весов)

            # Накапливаем статистику шага
            epoch_loss += losses.item()
            loop.set_postfix(loss=losses.item())

        # Считаем среднюю ошибку за всю эпоху
        avg_loss = epoch_loss / len(train_loader)
        print(f"📊 Эпоха {epoch} завершена | Средний комбинированный Loss: {avg_loss:.4f}")

        # ------------------------------------------
        # 5. Сохранение весов (Валидация по минимальному лоссу)
        # ------------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = "weights/detector_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✨ Ошибка снизилась! Веса детектора сохранены в: {save_path}")


if __name__ == "__main__":
    train_detection()