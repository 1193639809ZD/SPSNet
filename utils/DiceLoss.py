from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, mask):
        assert predict.size() == mask.size(), "the size of predict and target must be equal."
        bs = predict.size(0)

        pre = predict.reshape(bs, -1)
        tar = mask.reshape(bs, -1)
        # dice_coeff
        score = 2 * ((pre * tar).sum(1) + self.epsilon) / ((pre + tar).sum(1) + self.epsilon)
        score = 1 - score.sum() / bs
        return score
