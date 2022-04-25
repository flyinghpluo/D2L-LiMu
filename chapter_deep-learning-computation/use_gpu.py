#在jupyter notebook中查看gpu信息指令
# ! nvidia-smi

import torch
from torch import nn
torch.device('cpu'),torch.device('cuda'),torch.device('cuda:0'),torch.device('cuda:1')#cuda:0表示第1个gpu，cuda:1表示第2个gpu

torch.cuda.device_count()#查询gpu个数


def get_gpu(i=0):
    #如果存在，则返回gpu(i)，否则返回cpu()
    if torch.cuda.device_count()>= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
#返回所有可用的GPU，如果没有GPU，则返回[cpu(),]
def get_all_gpu():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')] #返回值为列表形式

get_gpu(),get_gpu(10),get_all_gpu()

X = torch.tensor([1,2,3])
X.device #查询张量所在的设备

Y = torch.tensor([1,2,3],device = get_gpu())#在GPU上面创建张量
#Y = torch.tensor([1,2,3],device = torch.device('cuda')) 与上面代码等价
Y.device,Y

Y1 = torch.rand(2, 3, device=get_gpu(0))
Y1

X_clone = X.cuda(0)#将X复制到gpu0上面
X_clone

X.cuda(0) + Y ,X ,X_clone+Y #X.cuda(0)并没有把X所在的cpu转移到gpu上面，而是相当于将X复制了一遍并转移到gpu上面，而没有改变X原先所在的设备

Y.cuda(0) is Y #Y已经存储在gpu0上，再次调用cuda(0)并不会再次复制一遍，分配新内存，而是直接返回Y

net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=get_gpu())
net(Y)#将模型参数放在gpu上面，同时也要保证输入数据也在gpu上面

net[0].weight.data.device #查看模型参数存储在哪个设备上面

