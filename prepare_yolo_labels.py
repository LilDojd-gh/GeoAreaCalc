import os
import cv2
import numpy as np


def convert_masks_to_yolo():
    mask_dir = "data/buildings_dataset/masks/"
    labels_out_dir = "data/vehicles_dataset/labels/"

    os.makedirs(labels_out_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
    print(f"Начинаем конвертацию {len(mask_files)} масок...")

    cars_found_total = 0
    files_with_cars = 0

    for mask_name in mask_files:
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        h, w = mask.shape
        # Выделяем только автомобили (класс 6 в RescueNet)
        vehicle_pixels = (mask == 6).astype(np.uint8) * 255
        contours, _ = cv2.findContours(vehicle_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Жестко отрезаем _lab и расширения, чтобы имя txt совпало с jpg картинки
        base_name = mask_name.replace('_lab.png', '').replace('_lab.jpg', '').replace('.png', '').replace('.jpg', '')
        txt_name = base_name + '.txt'

        txt_path = os.path.join(labels_out_dir, txt_name)

        if len(contours) > 0:
            files_with_cars += 1
            with open(txt_path, 'w') as f:
                for cnt in contours:
                    x, y, box_w, box_h = cv2.boundingRect(cnt)

                    if box_w < 5 or box_h < 5:
                        continue

                    x_center = (x + box_w / 2.0) / float(w)
                    y_center = (y + box_h / 2.0) / float(h)
                    norm_w = box_w / float(w)
                    norm_h = box_h / float(h)

                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    cars_found_total += 1

    print(f"Готово! Найдено {cars_found_total} машин на {files_with_cars} снимках.")
    print("Текстовые файлы сохранены в:", labels_out_dir)


if __name__ == "__main__":
    convert_masks_to_yolo()
