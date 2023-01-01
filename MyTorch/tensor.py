import numpy as np


__all__ = ['Tensor']


# Класс тензора - основной объект фреймфорка.
# Numpy данные с динамическим вычислительным графом 
# и обратным распространением градиентов (backward).
class Tensor (object):

    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        
        # Данные
        self.data = np.array(data)
        # Включено ли обратное распространение градиентов
        self.autograd = autograd
        # Градиенты
        self.grad = None

        # Для каждого объекта будем иметь свой id
        if(id is None):
            self.id = np.random.randint(0,1000000000)
        else:
            self.id = id
        
        # Запишем предков и операцию создания данного тензора
        self.creators = creators
        self.creation_op = creation_op
        
        self.children = {}
        
        # Пропишем этот тензор как потомка в тезорах-предках
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1


    # Функция проверяет, что в данном тензоре учтены градиенты всех его потомков.
    # Без этого нельзя продолжать распространять градиенты дальше вглубь. 
    def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            if(cnt != 0):
                return False
        return True 
        

    # Функция обратного распространения градиентов
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
 
            # Для вызова Tensor.backward()
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            # Для учета всех потомков
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backprop more than once")
                else:
                    print(grad_origin)
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
            
            # Градиентам не нужны градиенты
            assert grad.autograd == False
            
            # Отправляем градиенты предкам
            if self.creators is not None and\
               (self.all_children_grads_accounted_for() or grad_origin is None):

                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                if(self.creation_op == "sub"):
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if(self.creation_op == "mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new , self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)                    
                    
                # Матричное умножение
                if(self.creation_op == "mm"):
                    c0 = self.creators[0]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)

                    c1 = self.creators[1]
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)
                    
                if(self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())

                # Сумирование по оси (для градиентов надо наоборот расширить по оси)
                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim, self.creators[0].data.shape[dim]))

                # Расширение по оси (для градиентов надо сумировать по оси)
                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                
                if(self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__())
                
                # Для сигмоиды    
                if(self.creation_op == "sigmoid"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))
                
                # Для tanh
                if(self.creation_op == "tanh"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if(self.creation_op == "cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

                # Для индексирования
                if self.creation_op == 'getitem':
                    new_grad = np.zeros_like(self.creators[0].data)
                    new_grad[self.getitem_index] = self.grad.data
                    self.creators[0].backward(Tensor(new_grad))

                # Для функции reshape (как в numpy)
                if self.creation_op == 'reshape':
                    new_grad = self.grad.data.reshape(self.prev_shape)
                    self.creators[0].backward(Tensor(new_grad))

                    
    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="add")
        return Tensor(self.data + other.data)


    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    
    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="mul")
        return Tensor(self.data * other.data)    


    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    

    def expand(self, dim,copies):

        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)
    

    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        
        return Tensor(self.data.transpose())
    

    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self,x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))
    
    
    def sigmoid(self):
        if(self.autograd):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    
    def tanh(self):
        if(self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))
    
    
    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        return softmax_output
    

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t),-1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()
    
        if(self.autograd):
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)


    def __getitem__(self, index):
        if isinstance(index, slice):
            index = index
        else:
            index = index.data

        data = self.data[index]
        
        if self.autograd:
            new = Tensor(data,
                         autograd=True,
                         creators=[self],
                         creation_op="getitem"
                        )
            new.getitem_index = index
            return new
        else:
            return Tensor(data)


    def reshape(self, *args):
        shape = self.data.shape
        data = self.data.reshape(*args)

        if self.autograd:
            new = Tensor(data,
                         autograd=True,
                         creators=[self],
                         creation_op="reshape"
                        )
            new.prev_shape = shape
            return new
        else:
            return Tensor(data)


    def __repr__(self):
        return str(self.data.__repr__())


    def __str__(self):
        return str(self.data.__str__())  