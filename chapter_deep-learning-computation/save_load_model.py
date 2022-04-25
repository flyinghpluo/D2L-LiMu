import torch
from torch import nn
from torch.nn import functional as F
x = torch.arange(4)

#1.1 存储加载一个张量
torch.save(x,"x.pth")#存储一个张量
x2 = torch.load("x.pth")#从x.pth文件把存储的张量读取出来
x2

#1.2 存储加载一个张量列表
y = torch.zeros(4)
torch.save([x,y],"x2.pth")#存储一个张量列表，然后把它们读回内存
y2 = torch.load("x2.pth")
y2

#1.3 存储加载一个字典
mydict = {"x":x,"y":y}
torch.save(mydict,"mydict.pth")#存储一个字典，然后把它们加载出来
mydict2 = torch.load("mydict.pth")
mydict2

#1.4 存储加载模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(),"parameters.pth")#将模型所有参数都保存在parameters.pth文件中
clone = MLP()#在加载模型参数前需要实例化模型，在实例化过程中模型参数也会进行初始化
clone.load_state_dict(torch.load("parameters.pth"))#加载文件中存储的参数，然后将模型开始初始化后的参数都替换掉
clone

Y_clone = clone(X)
Y_clone == Y #由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同