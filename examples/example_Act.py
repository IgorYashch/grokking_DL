# Чтобы делать нормальный импорт прямо из этой папки
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Протестируем функции активации и лосс 
import numpy as np 
from MyTorch import Tensor
from MyTorch import nn
from MyTorch.optim import SGD

np.random.seed(0)

data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

model = nn.Sequential([nn.Linear(2,3), nn.Tanh(), nn.Linear(3,1), nn.Sigmoid()])

criterion = nn.MSELoss()

optim = SGD(parameters=model.get_parameters(), alpha=5)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()

    print(loss)