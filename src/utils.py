import torch
import cv2
import numpy as np


# ==========================================
# 1. Метрики качества для Сегментации
# ==========================================
def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """
    Рассчитывает метрики IoU (Intersection over Union) и Dice
    для оценки качества предсказанной маски зданий.
    """
    # Бинаризируем предсказания сети (все, что выше порога - считаем зданием)
    pred_binary = (pred_mask > threshold).float()
    true_mask = true_mask.float()

    # Считаем пересечение (intersection) и объединение (union)
    intersection = torch.sum(pred_binary * true_mask)
    union = torch.sum(pred_binary) + torch.sum(true_mask) - intersection

    # Чтобы избежать ошибки деления на ноль, добавляем epsilon (1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2.0 * intersection + 1e-6) / (torch.sum(pred_binary) + torch.sum(true_mask) + 1e-6)

    return iou.item(), dice.item()


# ==========================================
# 2. Расчет Масштаба (GSD) и Площади
# ==========================================
def calculate_gsd_from_cars(bboxes, real_car_length_m=4.5):
    """
    Рассчитывает GSD (Ground Sample Distance - метров в одном пикселе)
    на основе найденных детектором автомобилей.

    bboxes: массив рамок машин формата [xmin, ymin, xmax, ymax]
    real_car_length_m: средняя длина автомобиля в метрах (по умолчанию 4.5м)
    """
    if len(bboxes) == 0:
        # Если машин нет, возвращаем None (масштаб определить невозможно)
        return None

    lengths_in_pixels = []

    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # Длина машины на фото — это большая сторона рамки (bounding box)
        car_length_px = max(width, height)
        lengths_in_pixels.append(car_length_px)

    # Берем медианное значение, чтобы отсеять возможные выбросы детектора
    # (например, длинные грузовики или ошибочно захваченные мелкие объекты)
    median_length_px = np.median(lengths_in_pixels)

    # Вычисляем масштаб: сколько метров в одном пикселе
    gsd = real_car_length_m / median_length_px

    return gsd


def calculate_buildings_area(pred_mask, gsd, threshold=0.5):
    """
    Считает общую площадь застройки в квадратных метрах (базовый метод).
    """
    if gsd is None:
        raise ValueError("Невозможно рассчитать площадь: неизвестен масштаб (GSD).")

    # Считаем количество пикселей, которые модель определила как здания
    pred_binary = (pred_mask > threshold).numpy()
    total_pixels = np.sum(pred_binary)

    # Площадь одного пикселя = GSD * GSD
    # Общая площадь = количество пикселей * площадь одного пикселя
    area_sqm = total_pixels * (gsd ** 2)

    return area_sqm


def calculate_individual_areas(pred_mask, gsd, threshold=0.5):
    """
    Находит отдельные здания на бинарной маске семантической сегментации
    и считает площадь каждого в квадратных метрах (трюк с OpenCV).
    """
    if gsd is None:
        raise ValueError("Невозможно рассчитать площадь: неизвестен масштаб (GSD).")

    # Бинаризируем маску и переводим в формат uint8 (требование OpenCV)
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255

    # Находим связные компоненты (отдельные здания)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    individual_areas_sqm = []

    # Проходим по всем найденным объектам (начиная с 1, так как 0 - это фон)
    for i in range(1, num_labels):
        # Достаем площадь конкретного здания в пикселях из статистики
        area_px = stats[i, cv2.CC_STAT_AREA]

        # Переводим в квадратные метры
        area_sqm = area_px * (gsd ** 2)

        # Фильтруем случайный мусор-артефакты (считаем зданием то, что больше 10 кв.м)
        if area_sqm > 10.0:
            individual_areas_sqm.append(round(area_sqm, 2))

    total_area_sqm = sum(individual_areas_sqm)

    return total_area_sqm, individual_areas_sqm