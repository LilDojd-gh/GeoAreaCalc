import torch
import torch.nn as nn


# ==========================================
# 1. Кастомный Dice Loss
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Инициализация Dice Loss.
        smooth: константа для предотвращения ошибки деления на ноль,
                если на снимке (и в предсказании) вообще нет зданий.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: предсказания модели (вероятности от 0 до 1, пропущенные через Sigmoid)
        # targets: истинные бинарные маски (1 - здание, 0 - фон)

        # Вытягиваем тензоры в одномерный массив (flatten),
        # чтобы считать пересечение сразу по всему батчу картинок
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Считаем числитель (пересечение) и знаменатель
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # Лосс должен уменьшаться в процессе обучения, поэтому вычитаем метрику из единицы
        return 1.0 - dice_score


# ==========================================
# 2. Комбинированный BCE + Dice Loss
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        """
        Комбинация классической бинарной кросс-энтропии (BCE) и Dice Loss.
        weight_bce: вес для BCE Loss.
        weight_dice: вес для Dice Loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()  # Встроенный BCE Loss от PyTorch
        self.dice = DiceLoss()  # Наш кастомный Dice Loss
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        # Считаем обе ошибки
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        # Складываем с заданными весами
        total_loss = (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss)

        return total_loss