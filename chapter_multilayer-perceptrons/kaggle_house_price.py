import hashlib
import os.path
import tarfile
import zipfile

import d2l.torch
import numpy
import requests
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, log

'''
完整训练模型过程：下载数据集 -> 对数据集进行预处理（将数值数据进行标准化(均值0，方差1)处理，将字符串数据进行one-hot编码，对缺失数据进行处理）->加载数据集，需要将数据集转换为tensor格式
-> 定义模型，loss函数，优化器 ->进行 K折交叉验证 ，根据平均训练误差和平均验证误差大小从而选择比较好的模型(结构)以及比较好的超参数(epochs,learning_rate,weight_decay,batch_size等)
-> 根据选好后的模型和超参数再对整个训练集进行全部训练，此时不再把训练集分出一部分给验证集，得到最后训练好的模型（指最后得到的是训练好后的模型参数）-> 将训练后的模型用于真正测试集上面进行预测
'''
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#下载数据集，并返回下载文件位置，如果数据集已经下载过且shal相同，则不再重复进行下载
def download(name,cache_dir = os.path.join('../data','kaggle_house_data')):
    assert name in DATA_HUB,f"{name}不存在于{DATA_HUB}"
    os.makedirs(cache_dir,exist_ok=True)#创建文件夹，exist_ok=True表示重复创建,并覆盖之前的文件夹
    url,shal_hash = DATA_HUB[name]
    fname = os.path.join(cache_dir,url.split('/')[-1])
    if os.path.exists(fname):
        shal = hashlib.sha1()
        with open(fname,'rb') as f :
            while(True):
                data = f.read(1048576)
                if not data :
                    break
                shal.update(data)
            #判断shal是否相同，相同则不再重新下载数据集
            if shal.hexdigest() == shal_hash :
                return fname
    else:
        print(f'正在从{url}下载{fname}数据集....')
        #从网上下载数据集
        data_online = requests.get(url,stream=True,verify=True)
        with open(fname,'wb') as f :
            f.write(data_online.content)
        return fname
#下载并解压缩zip,tar,gz文件
def download_extract(name,folder=None):
    fname = download(name)
    #fname = 'D:/Codes/Codes/PycharmCodes/PytorchCodes/D2L-LiMu/train.zip'
    base_dir = os.path.dirname(fname)
    print("base_dir==", base_dir)
    data_dir,ext = os.path.splitext(fname)#ext为文件后缀名
    print("data_dir==",data_dir)
    print("ext==", ext)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname,'r')
    elif ext in ('.tar','.gz'):
        fp = tarfile.open(fname,'r')
    else:
        assert False,'只有zip和tar文件才能解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir,folder) if folder else data_dir
#从DATA_HUB中下载所有的数据集
def download_all():
    for name in DATA_HUB:
        download(name)
fname_train = download("kaggle_house_train")
fname_test = download('kaggle_house_test')
train_data = pd.read_csv(fname_train) #pandas读取csv文件，train_data数据类型为：<class 'pandas.core.frame.DataFrame'>
print("train_data.type =  ",type(train_data))
test_data = pd.read_csv(fname_test)
print("test_data.shape = ",test_data.shape)
print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]]) #打印训练数据集前四行里面前四列和后面三列，train_data数据类型为：<class 'pandas.core.frame.DataFrame'>
'''
pandas.concat()：pandas拼接函数
将训练集和测试集拼接在一起，能更方便用于一起进行数据预处理，拼接时去掉了训练集和测试集的第一列（编号id列）和训练集SalePrice列，
因为编号id对训练数据没有作用，不能把编号id加入进来，如果加入反而可能会让训练模型时记住这个编号id对应的SalePrice,从而时测试数据集时变得更糟糕，
去掉训练集SalePrice列，是因为SalePrice列是标签label列，必须去掉这一列，如果加入就会使训练模型提前知道labels，从而训练只会关注这一列，
因此代码train_data.iloc[:,1:-1]中-1必须含有（重要），而由于测试集中没有SalePrice列，因此不用去掉，因此代码中test_data.iloc[:,1:]中没有含有-1
'''
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
print(all_features.iloc[0:4,[0,1,2,3,-3,-2,-1]])
print("all_features.type = ",type(all_features))
#将拼接后的数据集中所有列的值不是字符串的列选出来，并对这些数值列进行数值标准化，将所有特征放在同一个共同的尺度下，通过将数值特征重新缩放到零均值和单位方差来标准化数据，从而避免数值过大时带来模型对参数求导时导数过大的现象，可能会出现导数达到无穷大的后果
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index #将数据集中数值列名称选出来
all_features[numeric_features] = all_features[numeric_features].apply(lambda x : (x-x.mean())/x.std()) #根据选出的数值列名对这些列进行数值特征标准化
all_features[numeric_features] = all_features[numeric_features].fillna(0) #将所有缺失的值替换为相应特征的平均值
print(all_features.iloc[0:4,[0,1,2,3,-3,-2,-1]])
#处理字符串特征值，采用one-hot编码方式。
#例如，“MSZoning”包含值“RL”和“Rm”。 我们将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。
#根据独热编码，如果“MSZoning”的原始值为“RL”， 则：“MSZoning_RL”为1，“MSZoning_RM”为0。 pandas软件包会自动为我们实现这一点
all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.iloc[:4,:])
n_train = train_data.shape[0] #得到训练数据集的个数，即行数
#预处理完后，将训练集和测试集分开并转换成tensor类型
#通过values属性，可以从pandas格式中提取NumPy格式，并将其转换为张量表示用于训练
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float32)
train_features = train_features.to(device)
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float32)
test_features = test_features.to(device)
#获取训练集的labels
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)
train_labels = train_labels.to(device)
loss = nn.MSELoss().to(device)#定义均方误差
in_features = train_features.shape[1]#得到训练集每个数据的特征个数，也即是列数==特征数
'''
定义模型网络：
首先，我们训练一个带有损失平方的线性模型。 显然线性模型很简单，实际中模型比这个很复杂，
但线性模型提供了一种健全性检查， 以查看数据中是否存在有意义的信息。
 如果我们在这里不能做得比随机猜测更好，那么我们很可能存在数据处理错误。 
 如果一切顺利，线性模型将作为基线（baseline）模型， 让我们直观地知道最好的模型有超出简单的模型多少。
 将线性模型作为基准，看其他复杂的模型比这个线性模型好多少或者坏多少
'''
# def get_net():
#     net = nn.Sequential(nn.Linear(in_features,1))#只包含一层线性层
#     return net
def get_net():
    net = nn.Sequential(nn.Linear(in_features,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1)).to(device)#只包含一层线性层
    return net
def log_rmse(net,features,labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1,因为对趋近于0取导数时会趋于负无穷大
    clipped_preds = torch.clamp(net(features),1,float('inf'))
    # 对预测结果和真实labels取对数，比较他们的相对误差y^/y大于1或者小于1，再取对数判断他们的误差大小，即log(y^/y)=log(y^)-log(y)
    rmse_loss = torch.sqrt(loss(log(clipped_preds),log(labels))).to(device)
    return rmse_loss.item()
# 训练模型
def train(net,train_datas,train_labels,test_datas,test_labels,batch_size,epochs,learning_rate,weight_decay):
    train_ls = []
    test_ls = []
    train_iter = d2l.torch.load_array((train_datas,train_labels),batch_size)#加载数据
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)#Adam优化算法，对初始学习率不那么敏感
    for epoch in range(epochs):
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            l = loss(net(X),y) #计算预测值和真实值损失大小，数据集损失平方和再求均值（均方误差）
            l.backward() #计算梯度
            optimizer.step() #更新参数
        # 模型训练时使用均方误差计算，当模型训练完后计算损失时使用rmse()相对损失计算，
        # 因为如果训练时使用相对损失计算时由于有开根号，log，求导数时会很复杂，很容易造成梯度爆炸或者梯度消失，因此训练模型时采用均方误差来进行计算
        train_ls.append(log_rmse(net,train_datas,train_labels))#计算每轮训练完后在训练集上面的损失
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_datas,test_labels))#计算每轮训练完后在验证集上面的损失
    return train_ls,test_ls
#采用k折交叉验证，有助于模型选择和超参数选择，选择第 𝑖 个切片作为验证数据，其余部分作为训练数据
def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0] // k #每个切片个数，注意训练集不能整除情况
    X_train,y_train = None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part = X[idx,:],y[idx]
        if j == i:
            X_valid,y_valid = X_part,y_part #选择第 𝑖 个切片作为验证数据
        else:
            if X_train == None:
                X_train = X_part
                y_train = y_part
            else:
                X_train = torch.cat([X_train,X_part],0) #其余部分作为训练数据
                y_train = torch.cat([y_train,y_part],0)
    return X_train,y_train,X_valid,y_valid
#在K折交叉验证中训练K次后，返回训练和验证误差的平均值，来表示对这个数据集的训练集和验证集误差大小是多少，有可能模型训练和验证时对某一个切片数据集比较敏感，导致误差小，另一个切片数据集不敏感，导致误差变大
def k_fold(k,train_datas,train_labels,batch_size,epochs,learning_rate,weight_decay):
    train_ls_sum = 0
    valid_ls_sum = 0
   # net = get_net()
    for i in range(k):
        # 注意：(重要) 每一次进行循环时都要重新初始化网络模型，重新对网络进行训练，不能使用上一次训练已经用于训练的模型，因为每一折表明训练集和验证集不同，
        # 且如果使用上一次循环训练后的网络来训练和验证这一折的数据集时，由于模型已经在上一次循环中看过部分训练集和验证集时，从而导致验证集误差变小，导致模型预测不准确，因此会使模型在真正测试集上面预测时误差会偏大
        net = get_net() #不能把模型初始化放在第157行代码处(循环外面)进行初始化，必须放在循环内进行初始化 ----重要
        data = get_k_fold_data(k,i,train_datas,train_labels)
        #在i折上对数据集训练epoch轮得出的对当前训练集和验证集上面的误差大小，比如将第一个切片作为验证集，其他数据作为训练集，则放入模型训练中，训练epoch轮后得出在每轮训练集和验证集上面的误差大小
        train_ls,valid_ls = train(net,*data,batch_size=batch_size,epochs=epochs,learning_rate=learning_rate,weight_decay=weight_decay)
        train_ls_sum += train_ls[-1] #得到除开第i个切片验证集以外的训练数据集在最后一轮的误差大小
        valid_ls_sum += valid_ls[-1]#得到第i个切片验证集在最后一轮的误差大小
        if i == 1:
            d2l.torch.plot(list(range(1, epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, epochs],
                     legend=['train', 'valid'], yscale='log')
            plt.show()
        print(f'{i+1}折，训练log_rmse{float(train_ls[-1]):f},验证log_rmse{float(valid_ls[-1]):f}')
        # print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
        #       f'验证log rmse{float(valid_ls[-1]):f}')
    return train_ls_sum / k,valid_ls_sum / k
#根据K折交叉验证得到平均训练误差和平均验证误差，从而判断当前模型和超参数是否好坏，从而进行模型和超参数选择
k,num_epochs,lr,weight_decay,batch_size = 5,1000,0.3,25000,64
train_loss,valid_loss = k_fold(k,train_features,train_labels,batch_size,num_epochs,lr,weight_decay)
print(f'{k}折，平均训练log_rms{float(train_loss):f},平均验证log_rmse{float(valid_loss):f}')
#根据选择出比较好的模型和超参数后再对训练集进行全部用于训练，此时不用在训练集上面分出一部分作为验证集，将训练好后的模型包含模型里面的参数用于真正的测试集上面验证
def train_pred(train_features,train_labels,test_features,test_data,batch_size,epochs,learning_rate,weight_decay):
    net = get_net()
    #将训练集全部用于训练，得到最后训练好的模型
    train_ls,_ = train(net,train_features,train_labels,None,None,batch_size=batch_size,epochs=epochs,learning_rate=learning_rate,weight_decay=weight_decay)
    d2l.torch.plot(numpy.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    #在matlib上面显示出来
    #plt.show()
    print("训练完全部数据后的loss = ",train_ls[-1])
    #将训练好的模型用于测试集上面进行预测得到测试集上面每个房子价格
    pred_price = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(pred_price.reshape(1,-1)[0])
    #使用pandas.cancat()函数进行列拼接，axis=1表示列拼接
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    #将拼接结果输出到csv文件中，submission类型为：<class 'pandas.core.frame.DataFrame'>
    submission.to_csv("../data/kaggle_house_data/sumission",index=False)
train_pred(train_features,train_labels,test_features,test_data,batch_size,num_epochs,lr,weight_decay)




