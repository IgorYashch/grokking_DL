# Чтобы делать нормальный импорт прямо из этой папки
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Протестируем класс Linear и последовательность слоев Sequential
import numpy as np
from MyTorch.tensor import Tensor
from MyTorch.nn import Sequential, Linear
from MyTorch.optim import SGD


np.random.seed(0)

# изначальные данные (размер 4x2)
data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)

# что в итоге хотим получить (размер 4x1)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

# Последовательность линейных слоев
model = Sequential([Linear(2, 3),
                    Linear(3, 1)
                   ])

# Оптимизатор
optim = SGD(model.get_parameters(), alpha=0.05)


for i in range(20):
    pred = model.forward(data)
    
    loss = ((pred - target) * (pred - target)).sum(0)
    
    loss.backward(Tensor(np.ones_like(loss.data)))
    
    optim.step()
    
    print(loss)