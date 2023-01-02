from .module import Module


__all__ = ['Tanh', 'Sigmoid', 'ReLU']



class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input.relu()


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()



class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()