import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(smooth)

    def forward(self, y_pred, y_true):
        y_pred = F.sigmoid(y_pred)
        dl = self.dice_loss(y_pred, y_true)
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        return dl + bce
