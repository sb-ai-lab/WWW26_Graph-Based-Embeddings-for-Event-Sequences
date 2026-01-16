import torch.nn as nn

class ConvexCombinationLoss(nn.Module):
    def __init__(self, loss1, loss2, alpha=0.5):
        super().__init__()
        assert 0 <= alpha <= 1
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

    def forward(self, input, target):
        return self.alpha * self.loss1(input, target) + (1 - self.alpha) * self.loss2(input, target)