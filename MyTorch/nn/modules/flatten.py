from .module import Module


__all__ = ["Flatten"]


# просто передедываем в [BATCH_SIZE, -1]
class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)
