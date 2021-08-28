import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementations based on https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

ALPHA = 0.5
BETA = 0.5
GAMMA = 2.0


class DiceLoss(nn.Module):
    # The Dice coefficient, or Dice-Sørensen coefficient, is a common metric
    # for pixel segmentation that can also be modified to act as a loss
    # function
    def __init__(self, smooth: float = 1.0, apply_sigmoid: bool = False):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.size() == y_true.size()
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc


class DiceBCELoss(nn.Module):
    # This loss combines Dice loss with the standard binary cross-entropy (BCE)
    # loss that is generally the default for segmentation models. Combining the
    # two methods allows for some diversity in the loss, while benefitting from
    # the stability of BCE.
    def __init__(self, smooth: float = 1.0, apply_sigmoid: bool = False):
        super().__init__()
        self.dice_loss = DiceLoss(smooth)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)

        dl = self.dice_loss(y_pred, y_true)
        bce = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
        return dl + bce


class IoULoss(nn.Module):
    # The IoU metric, or Jaccard Index, is similar to the Dice metric and is
    # calculated as the ratio between the overlap of the positive instances
    # between two sets, and their mutual combined values.
    # Like the Dice metric, it is a common means of evaluating the performance
    # of pixel segmentation models.
    def __init__(self, smooth: float = 1.0, apply_sigmoid: bool = False):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    # Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 as
    # a means of combatting extremely imbalanced datasets where positive cases
    # were relatively rare. Their paper "Focal Loss for Dense Object Detection"
    # is retrievable here: https://arxiv.org/abs/1708.02002. In practice, the
    # researchers used an alpha-modified version of the function so I have
    # included it in this implementation.
    def __init__(
        self,
        apply_sigmoid: bool = False,
        smooth: float = 1.0,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
    ):
        super(FocalLoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        if self.apply_sigmoid:
            y_pred = F.sigmoid(y_pred)

        # flatten label and prediction tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1.0 - BCE_EXP) ** self.gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    # This loss was introduced in "Tversky loss function for image
    # segmentation using 3D fully convolutional deep networks", retrievable here:
    # https://arxiv.org/abs/1706.05721. It was designed to optimise segmentation on
    # imbalanced medical datasets by utilising constants that can adjust how
    # harshly different types of error are penalised in the loss function. From the
    # paper: ... in the case of α=β=0.5 the Tversky index simplifies to be the same
    # as the Dice coefficient, which is also equal to the F1 score. With α=β=1,
    # Equation 2 produces Tanimoto coefficient, and setting α+β=1 produces the set
    # of Fβ scores. Larger βs weigh recall higher than precision (by placing more
    # emphasis on false negatives). To summarise, this loss function is weighted by
    # the constants 'alpha' and 'beta' that penalise false positives and false
    # negatives respectively to a higher degree in the loss function as their
    # value is increased. The beta constant in particular has applications in
    # situations where models can obtain misleadingly positive performance via
    # highly conservative prediction. You may want to experiment with different
    # values to find the optimum. With alpha==beta==0.5, this loss becomes
    # equivalent to Dice Loss.
    def __init__(
        self,
        smooth: float = 1.0,
        apply_sigmoid: bool = False,
        alpha: float = ALPHA,
        beta: float = BETA,
    ):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - tversky
