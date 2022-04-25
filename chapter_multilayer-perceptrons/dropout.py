import torch
from d2l.torch import d2l
from torch import nn
from torch.nn import Linear

#该函数以dropout的概率丢弃张量输入X中的元素，重新缩放剩余部分：将剩余部分除以(1.0-dropou)。
def dropout_layer(X,dropout):
    #dropout范围在[0,1]之间
    assert 0<=dropout<=1
    #dropout = 1指把输出层所有元素都丢掉
    if dropout == 1:
        return torch.zeros_like(X)
    # dropout = 0指把输出层所有元素都保留，不丢掉
    if dropout ==0:
        return X
    #zero_one_X = torch.randn(X.shape)按照X尺寸维数大小按照正态分布随机生成0到1之间的元素所组成的矩阵，然后与dropout比较大小得到的bool值True,False，然后转换成浮点数，对应1,0值
    zero_one_X = torch.randn(X.shape)
    #print("zero_one_X = ",zero_one_X)
    mask = (zero_one_X > dropout).float()
    # mask = (torch.rand(X.shape) > dropout).float()#与上面两行代码等价
    #print("mask ：",mask)
    #mask矩阵和X矩阵对应元素相乘再除以(1-dropout)，保证使用dropout后保证每一层使用dropout后的输出层元素期望均值不变
    return mask * X /(1-dropout) #在Dropout模型实现过程中Dropout掉的元素其实是把这个元素变为0，然后继续输入到下一层中，输出层的维数仍然没有改变，仍然为256维
X = torch.arange(16,dtype=torch.float32).reshape(2,8)

#测试dropout_layer()函数
def test_dropout():
    print("X: ", X)
    print("dropout = 0", dropout_layer(X, dropout=0))
    print("dropout = 0.2", dropout_layer(X, dropout=0.2))
    print("dropout = 0.5", dropout_layer(X, dropout=0.5))
    print("dropout = 1", dropout_layer(X, dropout=1))
test_dropout()

num_inputs = 784
num_outputs = 10
num_hiddens1 = 256
num_hiddens2 = 256
dropout_1 = 0.2
dropout_2 = 0.5

class model(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_training = True):
        super(model, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.is_training = is_training
        self.linear1 = nn.Linear(num_inputs,num_hiddens1,bias=True)
        self.linear2 = nn.Linear(num_hiddens1,num_hiddens2,bias=True)
        self.linear3 = nn.Linear(num_hiddens2,num_outputs,bias=True)
        self.relu = nn.ReLU()

    def forward(self,X):
        X = X.reshape(-1,self.num_inputs)
        l1 = self.linear1(X)
        H1 = self.relu(l1)
        #is_training 为true时表明数据集此时为训练集，需要使用dropout，测试集测试时不使用dropout
        if self.is_training:
            H1 = dropout_layer(H1,dropout=dropout_1)
        l2 = self.linear2(H1)
        H2 = self.relu(l2)
        if self.is_training:
            H2 = dropout_layer(H2,dropout=dropout_2)
        out = self.linear3(H2)
        return out
def network_train(net):
    num_epoch = 10
    lr = 0.5
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    #分类用CrossEntropyLoss()损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    #使用SGD梯度下降优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, trainer)

#从零到一代码层实现dropout
def dropout_scratch():
    net = model(num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True)
    print("net : ", net)
    network_train(net)


#调用pytorch框架简洁实现
def dropout_concise():
    net = nn.Sequential(nn.Flatten(),
        nn.Linear(num_inputs,num_hiddens1,bias=True),
        nn.ReLU(),
        #使用pytorch框架简洁实现即直接在ReLU()后面加入dropout()层
        nn.Dropout(dropout_1),#在Dropout模型实现过程中Dropout掉的元素其实是把这个元素变为0，然后继续输入到下一层中，输出层的维数仍然没有改变，仍然为256维
        nn.Linear(num_hiddens1,num_hiddens2,bias=True),
        nn.ReLU(),
        nn.Dropout(dropout_2),
        nn.Linear(num_hiddens2,num_outputs)
    )
    #初始化每一个线性层的参数weights,使所有初始weights都在均值0，标准差0.01的正态分布中
    net.apply(init_weights)
    network_train(net)


def init_weights(model):
    if(type(model) == nn.Linear):
        nn.init.normal_(model.weight,std = 0.01)

#dropout_scratch()
dropout_concise()


