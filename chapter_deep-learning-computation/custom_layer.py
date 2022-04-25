#自定义层：目前如果存在一个在深度学习框架中还不存在的层。 在这些情况下，你必须构建自定义层。
import torch
import torch.nn.functional as F
from torch import nn

#构造一个没有任何参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()#__init__()函数不进行任何初始化操作

    def forward(self, X):
        return X - X.mean()#对X中每一个元素减去均值

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
'''
输出结果：
tensor([-2., -1.,  0.,  1.,  2.])
'''

#将自定义层作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean() #当给网络发送随机数据后，检查输出Y均值是否为0。 由于我们处理的是浮点数，因为存储精度的原因，我们仍然可能会看到一个非常小的非零数。
'''
输出结果：
tensor(1.8626e-09, grad_fn=<MeanBackward0>)
'''

#自定义一个带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))#采用随机初始化参数，并由Parameter()函数包裹起来，目的是给这个参数一个名字标识
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

#实例化MyLinear类并访问其模型权重参数
linear = MyLinear(5, 3)#输入5维，输出3维的层
linear.weight #随机初始化权重参数经过Parameter()函数包裹后的输出结果，多了一个"Parameter containing"名字标识
'''
输出结果：Parameter containing:
tensor([[ 0.3891, -0.4105, -0.7189],
        [-0.8655, -0.1785, -0.0529],
        [-0.1446,  1.7960,  0.1220],
        [ 0.4567,  0.0444, -0.5521],
        [ 0.8643, -1.2224,  0.5013]], requires_grad=True)
'''
#使用自定义层直接执行前向传播计算
linear(torch.rand(2, 5))
'''
输出结果：
tensor([[0.0000, 1.0663, 0.0194],
        [0.0000, 0.0000, 0.7687]])
'''


#使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))#Sequential()包含两个自定义层
net(torch.rand(2, 64))#给模型输入，得到输出
'''
输出结果：
tensor([[7.9190],
        [1.8329]])
'''


import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)

class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
net = MLP()
net(X)


class MySequential(nn.Module):

    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        print(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
            print(X)
        return X.sum()
net = FixedHiddenMLP()
net(X)

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)