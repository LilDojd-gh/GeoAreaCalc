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
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Инициализация датасета для сегментации зданий.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Получаем список файлов (названия картинок и масок должны совпадать)
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Загрузка картинки
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загрузка маски (в градациях серого)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Бинаризация: пиксели зданий (255) станут 1.0, фон (0) останется 0.0
        mask = (mask / 255.0).astype(np.float32)

        # Применение аугментаций
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        # Добавляем измерение канала для маски [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


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
        """
        Инициализация датасета для детекции машин.
        Ожидает разметку в формате YOLO (.txt файлы).
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = []
        labels = []

        # Парсинг YOLO-разметки
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        x_c, y_c, bw, bh = map(float, data[1:])

                        # Перевод в абсолютные координаты [xmin, ymin, xmax, ymax]
                        xmin = (x_c - bw / 2) * w
                        ymin = (y_c - bh / 2) * h
                        xmax = (x_c + bw / 2) * w
                        ymax = (y_c + bh / 2) * h

                        boxes.append([xmin, ymin, xmax, ymax])

                        # Faster R-CNN резервирует класс 0 под фон, сдвигаем метку на +1
                        labels.append(class_id + 1)

        # Обработка пустых снимков (без машин)
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        # Применение аугментаций с пересчетом рамок
        if self.transform is not None:
            augmentations = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = augmentations['image']
            boxes = np.array(augmentations['bboxes'], dtype=np.float32)
            labels = np.array(augmentations['class_labels'], dtype=np.int64)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Упаковка в словарь (требование torchvision)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return image, target


# Трансформации и аугментации для детекции
detection_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))