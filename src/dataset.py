import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==========================================
# 1. Датасет для Сегментации (Базовая задача)
# ==========================================
class SatelliteSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, building_classes=(2, 3, 4, 5)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.building_classes = building_classes

        self.valid_data = []
        self.cache = {}  # <--- СЛОВАРЬ ДЛЯ КЭШИРОВАНИЯ В RAM из за объема фото (4к) но их сжатия
                         # до 256х256 лучше всего сделать именно так для экономии времени на эпохах после 1

        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            base_name = os.path.splitext(img_name)[0]
            mask_png = os.path.join(mask_dir, base_name + '_lab.png')
            mask_jpg = os.path.join(mask_dir, base_name + '_lab.jpg')

            if os.path.exists(mask_png):
                self.valid_data.append((img_name, base_name + '_lab.png'))
            elif os.path.exists(mask_jpg):
                self.valid_data.append((img_name, base_name + '_lab.jpg'))

        print(f"✅ Загружено {len(self.valid_data)} валидных пар (картинка + маска)")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        # 1. ПРОВЕРЯЕМ КЭШ. Если картинка уже была считана, берем из RAM (мгновенно!)
        if idx in self.cache:
            image, binary_mask = self.cache[idx]

        # 2. ЕСЛИ В КЭШЕ НЕТ, читаем с диска (медленно, но только на 1-й эпохе)
        else:
            img_name, mask_name = self.valid_data[idx]
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                binary_mask = np.zeros((256, 256), dtype=np.float32)
            else:
                binary_mask = np.isin(mask, self.building_classes).astype(np.float32)

                # ЖЕСТКАЯ ОПТИМИЗАЦИЯ: Сжимаем тяжелое фото до 256x256 ПЕРЕД кэшированием,
                # чтобы не забить всю оперативку 4K-снимками
                image = cv2.resize(image, (256, 256))
                # Для маски используем INTER_NEAREST, чтобы не смазать контуры (оставить строго 0 и 1)
                binary_mask = cv2.resize(binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            # Записываем сжатые тензоры в оперативную память
            self.cache[idx] = (image, binary_mask)

        # 3. Применение аугментаций (повороты, флипы)
        # Аугментации применяются НА ЛЕТУ, поэтому каждую эпоху сеть видит чуть новые картинки
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=binary_mask)
            image = augmentations["image"]
            binary_mask = augmentations["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            binary_mask = torch.from_numpy(binary_mask).float()

        if binary_mask.ndim == 2:
            binary_mask = binary_mask.unsqueeze(0)

        return image, binary_mask


# Трансформации и аугментации для сегментации
train_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Rotate(limit=35, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])


# ==========================================
# 2. Датасет для Детекции (Альтернативная задача)
# ==========================================
class VehicleDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        # Кэш удален для экономии оперативной памяти (RAM)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Читаем картинку и разметку с диска КАЖДЫЙ РАЗ
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, txt_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5: continue

                    class_id, x_c, y_c, bw, bh = map(float, parts)

                    raw_xmin = (x_c - bw / 2) * w
                    raw_ymin = (y_c - bh / 2) * h
                    raw_xmax = (x_c + bw / 2) * w
                    raw_ymax = (y_c + bh / 2) * h

                    # Жесткая защита от выхода за границы
                    xmin = max(0.0, raw_xmin)
                    ymin = max(0.0, raw_ymin)
                    xmax = min(float(w), raw_xmax)
                    ymax = min(float(h), raw_ymax)

                    if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)  # 1 - класс "Автомобиль"

        # Аугментации и форматирование тензоров
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float() / 255.0

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return image, target


# Трансформации и аугментации для детекции
detection_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
