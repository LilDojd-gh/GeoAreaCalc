import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ==========================================
# 1. Архитектура Сегментации (U-Net)
# ==========================================
class UNetSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        Инициализация архитектуры U-Net.
        in_channels: 3 (RGB изображение)
        out_channels: 1 (Бинарная маска зданий)
        """
        super(UNetSegmentation, self).__init__()

        # Энкодер (сжатие пространства и извлечение признаков)
        self.enc1 = self._conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        # Боттлнек (наибольшее сжатие)
        self.bottleneck = self._conv_block(32, 64)

        # Декодер (расширение обратно до исходного размера)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # После конкатенации будет 32 (от upconv) + 32 (от enc2) = 64 канала. Ужимаем в 32.
        self.dec1 = self._conv_block(64, 32)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # После конкатенации будет 16 (от upconv) + 16 (от enc1) = 32 канала. Ужимаем в 16.
        self.dec2 = self._conv_block(32, 16)

        # Финальный выходной слой
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        """Вспомогательный блок из двух сверток и функций активации."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Проход по энкодеру со сбором данных для Skip Connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Проход через боттлнек
        b = self.bottleneck(self.pool2(e2))

        # Проход по декодеру с пробросом (конкатенацией) признаков
        d1 = self.upconv1(b)
        d1 = torch.cat((d1, e2), dim=1)
        d1 = self.dec1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec2(d2)

        # Финальная свертка
        out = self.final_conv(d2)

        # Возвращаем вероятности от 0.0 до 1.0 (важно для BCE Loss)
        return torch.sigmoid(out)


# ==========================================
# 2. Архитектура Детекции (Faster R-CNN)
# ==========================================
def get_vehicle_detection_model(num_classes=2):
    """
    Возвращает модель Faster R-CNN, настроенную на поиск объектов.
    num_classes: Количество классов. В нашем случае 2 (0 = фон, 1 = автомобиль).
    """
    # Загружаем предобученную модель на базе архитектуры ResNet50-FPN
    # weights='DEFAULT' загружает лучшие доступные веса, обученные на COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Получаем количество входных признаков для классификатора в "голове" сети
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Заменяем стандартную "голову" (которая была на 91 класс)
    # на новую, с нужным нам количеством классов
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model