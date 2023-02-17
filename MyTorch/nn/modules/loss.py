from .module import Module


__all__ = ["MSELoss", "CrossEntropyLoss"]


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)
