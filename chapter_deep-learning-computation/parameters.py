import torch
from torch import nn

#1.定义网络模型
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
y = torch.rand(2,1)
print("y = ",y)
print(net)
print(net[2].state_dict())

# 2.查看每一层网络权重参数：
print("bias.type = ",type(net[2].bias))#输出参数类型
print("bias = ",net[2].bias)#类型为Parameter
print("bias.data = ",net[2].bias.data)#类型为tensor格式
print("bias.data.item() = ",net[2].bias.data.item())#获取tensor里面元素值需要调用item()函数
print("weight = ",net[2].weight)

# 3.查看网络每一层的梯度：
print("查看计算梯度前的grad ： ")
print(net[2].weight.grad == None)
y = torch.rand(2,1)
loss = nn.MSELoss()
l = loss(net(X),y.reshape(2,-1))
l.backward()
print("查看计算梯度后的grad ： ")
print(net[2].weight.grad)

# 4.一次性查看网络所有参数：
print(*[(name,parameters.shape) for name,parameters in net[0].named_parameters()])
print(*[(name,parameters.shape) for name,parameters in net.named_parameters()])

print("=====访问网络所有参数：=====")
print(net.state_dict())#类型为有序键值对dict
print("======打印第三层bias:======")
print(net.state_dict()['2.bias'])


# 5.从嵌套块查看参数：
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block{i}",block1())#block{i}是对添加的模块的名字标识，如果名字相同则会进行覆盖
    return net
rgnet = nn.Sequential(block2(),block1())
print("打印rgnet网络结构：")
print(rgnet)
print(rgnet(X))
print("查看第一个主块中第二个子块中第一个层的bias值：")
print(rgnet[0][1][0].bias.data)

# 6.模型参数初始化：
## 6.1内置初始化：
print("网络手动初始化前的参数：")
print(net[0].weight.data)#第一层的weight
print(net[0].bias.data)#第一层的bias
def init_normal(m):
    if type(m) ==nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=1)#权重参数初始化为均值为0，方差为1的高斯随机变量，正态分布
        nn.init.zeros_(m.bias)#将网络bias初始化为0
net.apply(init_normal)#将网络所有层递归调用init_normal()函数，如果当前层为线性层，则将其weight,bias参数进行初始化
print("网络手动初始化后的参数：")
print(net[0].weight.data)#第一层的weight
print(net[0].bias.data)#第一层的bias

#将线性网络所有参数初始化为给定的常数，比如初始化为1
print("网络手动初始化前的参数：")
print(net[0].weight.data)#第一层的weight
print(net[0].bias.data)#第一层的bias
def init_constant(m):
    if type(m) ==nn.Linear:
        nn.init.constant_(m.weight,1)#权重参数初始化为1
        nn.init.zeros_(m.bias)#将网络bias初始化为0
net.apply(init_normal)#将网络所有层递归调用init_normal()函数，如果当前层为线性层，则将其weight,bias参数进行初始化
print("网络手动初始化后的参数：")
print(net[0].weight.data)#第一层的weight
print(net[0].bias.data)#第一层的bias


#对不同层应用不同的初始化方法，比如：使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_constant_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)
net[0].apply(init_xavier)#对第一层使用xavier初始化参数
net[2].apply(init_constant_42)#对第三层使用常量初始化参数
print(net[0].weight.data)
print(net[2].weight.data)

## 6.2 自定义初始化参数：
def my_init(m):
    if type(m) == nn.Linear:
        print("init",*[(name,parameters.shape) for name,parameters in m.named_parameters()][0])
        nn.init.uniform(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs()>=5
net.apply(my_init)
#直接对参数进行初始化
net[0].weight[:2]
net[0].weight.data[:]+=1
net[0].weight.data[0,0]=42
net[0].weight.data[0]


# 7.共享(绑定)网络层参数：在多个层共享参数（绑定参数）
shared_layer = nn.Linear(8,8)
#模型第三层和第五层都是同一个层shared_layer,都共享shared_layer层参数，指向（引用）shared_layer,当shared_layer参数发生改变时，该模型第三层和第五层参数也会跟着改变
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared_layer,nn.ReLU(),shared_layer,nn.ReLU(),nn.Linear(8,1))
net(X)
print(net[2].weight.data == net[4].weight.data)
net[2].weight.data[0,0] = 100 #当改变第三层参数时，第五层参数也会跟着改变
print(net[2].weight.data == net[4].weight.data)