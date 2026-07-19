import os
import cv2
import numpy as np


def convert_masks_to_yolo():
    # Пути к маскам RescueNet и куда сохранять текстовые файлы
    mask_dir = "data/buildings_dataset/masks/"
    labels_out_dir = "data/vehicles_dataset/labels/"

    # Создаем папку для txt файлов, если ее нет
    os.makedirs(labels_out_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    print(f"Начинаем конвертацию {len(mask_files)} масок...")

    cars_found_total = 0

    for mask_name in mask_files:
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h, w = mask.shape

        # Выделяем только пиксели автомобилей (значение 6)
        # Переводим в формат 255 (белый) для OpenCV
        vehicle_pixels = (mask == 6).astype(np.uint8) * 255

        # Находим связные области (отдельные машины)
        contours, _ = cv2.findContours(vehicle_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        txt_name = os.path.splitext(mask_name)[0] + '.txt'
        txt_path = os.path.join(labels_out_dir, txt_name)

        # Если машин на фото нет, txt файл просто не создастся или будет пустым
        # (наш датасет это обработает)
        if len(contours) > 0:
            with open(txt_path, 'w') as f:
                for cnt in contours:
                    # Получаем координаты рамки вокруг машины
                    x, y, box_w, box_h = cv2.boundingRect(cnt)

                    # Игнорируем мусор (пятна меньше 5x5 пикселей)
                    if box_w < 5 or box_h < 5:
                        continue

                    # Перевод в формат YOLO (нормализованные координаты от 0 до 1)
                    x_center = (x + box_w / 2.0) / float(w)
                    y_center = (y + box_h / 2.0) / float(h)
                    norm_w = box_w / float(w)
                    norm_h = box_h / float(h)

                    # Пишем в файл: класс 0 (машина), x_c, y_c, w, h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    cars_found_total += 1

    print(f"Готово! Извлечено {cars_found_total} Bounding Boxes автомобилей.")


if __name__ == "__main__":
    convert_masks_to_yolo()