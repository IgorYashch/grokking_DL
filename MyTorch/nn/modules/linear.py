import numpy as np
from .module import Module
from ...tensor import Tensor


__all__ = ["Linear"]


# Линейный слой
class Linear(Module):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()

        self.use_bias = bias

        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / (n_inputs))
        self.weight = Tensor(W, autograd=True)
        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)

        if self.use_bias:
            self.parameters.append(self.bias)

    def forward(self, input):
        if self.use_bias:
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        return input.mm(self.weight)
