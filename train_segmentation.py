import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Импортируем компоненты из нашего пакета src
from src import SatelliteSegmentationDataset, UNetSegmentation, BCEDiceLoss, calculate_metrics
from src.dataset import train_transform  # Импортируем набор аугментаций


def train_segmentation():
    # ------------------------------------------
    # 1. Настройки гиперпараметров
    # ------------------------------------------
    EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Запуск обучения U-Net на устройстве: {DEVICE}")

    # ------------------------------------------
    # 2. Подготовка пайплайна данных
    # ------------------------------------------
    # Пути к изображениям и маскам зданий
    train_image_dir = "data/buildings_dataset/images/"
    train_mask_dir = "data/buildings_dataset/masks/"

    # Инициализируем кастомный датасет с аугментациями Albumentations
    train_dataset = SatelliteSegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    # Оборачиваем в DataLoader для разбивки данных на батчи
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False
    )

    # ------------------------------------------
    # 3. Инициализация компонентов модели
    # ------------------------------------------
    model = UNetSegmentation(in_channels=3, out_channels=1).to(DEVICE)

    # Комбинированный лосс (BCE + Dice) против дисбаланса классов
    criterion = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    # Оптимизатор Adam для стабильного обновления весов
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Создаем папку для сохранения лучших результатов, если её нет
    os.makedirs("weights", exist_ok=True)
    best_iou = 0.0

    # ------------------------------------------
    # 4. Главный цикл обучения (Training Loop)
    # ------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()  # Переводим модель в режим обучения (активация Dropout/BatchNorm)
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0

        # Индикатор прогресса в консоли (tqdm)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for images, masks in loop:
            # Переносим входные тензоры на GPU/CPU
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # --- Forward Pass (Прямой проход) ---
            predictions = model(images)
            loss = criterion(predictions, masks)

            # --- Backward Pass (Обратное распространение ошибки) ---
            optimizer.zero_grad()  # Обнуляем старые градиенты, чтобы они не накапливались
            loss.backward()  # Считаем производные (градиенты) ошибки по весам
            optimizer.step()  # Корректируем веса сети

            # --- Сбор метрик качества ---
            # Отсекаем тензоры от графа вычислений (.detach()) для экономии памяти
            preds_cpu = predictions.detach().cpu()
            masks_cpu = masks.detach().cpu()

            iou, dice = calculate_metrics(preds_cpu, masks_cpu, threshold=0.5)

            # Накапливаем статистику шага
            epoch_loss += loss.item()
            epoch_iou += iou
            epoch_dice += dice

            # Динамически выводим текущие метрики в прогресс-бар
            loop.set_postfix(loss=loss.item(), iou=iou, dice=dice)

        # Считаем средние показатели за всю эпоху
        avg_loss = epoch_loss / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)

        print(f"📊 Эпоха {epoch} завершена | Средний Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f}")

        # ------------------------------------------
        # 5. Сохранение весов (Валидация по рекорду IoU)
        # ------------------------------------------
        if avg_iou > best_iou:
            best_iou = avg_iou
            save_path = "weights/unet_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Найден лучший результат! Веса сохранены в: {save_path}")


if __name__ == "__main__":
    train_segmentation()