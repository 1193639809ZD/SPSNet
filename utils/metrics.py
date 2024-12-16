from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
                pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):
    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
            input.long().sum().data.cpu()[0]
            + target.long().sum().data.cpu()[0]
            - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    batch_size = input.shape[0]

    input = input.reshape(batch_size, -1)
    target = target.reshape(batch_size, -1)

    intersection = (input * target).sum(1)

    loss = (2.0 * intersection + 1.0) / (input.sum(1) + target.sum(1) + 1.0)

    return loss.mean().item()
