# Базовый класс модуля (слоя)
# Все остальные наследуются от него

__all__ = ['Module']

class Module(object):
    def __init__(self):
        self.parameters = list()
        
    def get_parameters(self):
        return self.parameters