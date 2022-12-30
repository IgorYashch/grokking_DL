# Чтобы делать нормальный импорт прямо из этой папки
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Пример использования SGD
from MyTorch import Tensor
from MyTorch.optim import SGD
import numpy as np

np.random.seed(0)

# изначальные данные (размер 4x2)
data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)

# что в итоге хотим получить (размер 4x1)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

# список слоев с весами
weights_list = list()
weights_list.append(Tensor(np.random.rand(2,3), autograd=True))
weights_list.append(Tensor(np.random.rand(3,1), autograd=True))

# оптимизатор, который будет изменять веса из weights_list в зависимости от их градиента
optim = SGD(weights_list, alpha=0.1)

for iteration in range(20):
    # считаем предсказание
    pred = data.mm(weights_list[0]).mm(weights_list[1])
    # считаем ошибку
    loss = ((pred - target) * (pred - target)).sum(0)
    
    # считаем градиенты для динамического дерева
    loss.backward(Tensor(np.ones_like(loss.data)))
    # изменяем веса из weights_list в зависимости от их градиента
    optim.step()
    
    print(loss)
