# Чтобы делать нормальный импорт прямо из этой папки
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Протестируем слой Embedingа 
import numpy as np 
from MyTorch import Tensor
from MyTorch.nn import MSELoss, Linear, Sigmoid, Sequential, Embedding, Tanh
from MyTorch.optim import SGD

np.random.seed(0)


data = Tensor(np.array([1,2,1,2]), autograd=True)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)
embed = Embedding(5,3)
model = Sequential([embed, Tanh(), Linear(3,1), Sigmoid()])
criterion = MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=0.5)
for i in range(50):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)