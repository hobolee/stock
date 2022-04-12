import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
import time
from torch.utils.data import DataLoader, Dataset
import random
from collections import OrderedDict


class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            if type(module) is torch.nn.modules.rnn.LSTM:
                # input = input.view(-1, 30, 84*4)
                input, (h_n, c_n) = module(input)
                input = input[:, -1, :]
                # print('lstm', input.size())
            else:
                input = module(input)
                # print('other', input.size())
        return input

net = MySequential(
            nn.Conv3d(3, 16, (5, 7, 7), stride=1, padding=0), # in_channels, out_channels, kernel_size
            nn.BatchNorm3d(16),
            nn.Sigmoid(),
            nn.MaxPool3d(2, 2), # kernel_size, stride
            # nn.Conv3d(16, 64, 5),
            # nn.BatchNorm2d(64),
            # nn.Sigmoid(),
            # nn.MaxPool3d(2, 2),
            d2l.MyFlattenLayer(),
            nn.Linear(16*25*25, 480),
            nn.BatchNorm1d(13, 480),
            nn.Sigmoid(),
            nn.Linear(480, 84*4),
            nn.BatchNorm1d(13, 84*4),
            nn.Sigmoid(),
            #nn.Linear(84, 10)
            nn.LSTM(84*4, 1024, num_layers=1, batch_first=True),
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )


def train_ch5(net, batch_size, optimizer, device, num_epochs):
    global X, Y
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.MSELoss()
    batch_count = 0
    L = []
    for epoch in range(num_epochs):
        train_l_sum, start = 0.0, time.time()
        train_iter = data_iter_random(X[:, :, :, :], Y[:], 32, 30)
        for x, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(x)
            y_hat = y_hat.view(y_hat.shape[0])
            # print(y.size())
            # print(y_hat.size())

            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            batch_count += 1
            L.append(l)
        print('epoch %d, loss %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, time.time() - start))
        # x_test = X[1200:1230, :, :, :]
        # x_test = x_test.transpose(0, 1)
        # x_test = x_test.view(1, 3, 30, 57, 57)
        # print(net(x_test))
        # x_test = X[0:30, :, :, :]
        # x_test = x_test.transpose(0, 1)
        # x_test = x_test.view(1, 3, 30, 57, 57)
        # print(net(x_test))
        torch.save({'epoch': epoch / batch_count / batch_size,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': L},
                   './model1.pt')


global X, Y
X = torch.load("./X.pt").float()
Y = torch.load("./Y.pt").float()
X = X.view(2516, 3, 57, 57)
# print(Y.size())

# train_loader = DataLoader(X, batch_size=128, shuffle=False)
# test_loader = DataLoader(Y, batch_size=128, shuffle=False)


def data_iter_random(X, Y, batch_size, num_steps, device=None):
    #     print(X.size())
    num_examples = (len(Y) - num_steps)
    #     print('examples', num_examples)
    epoch_size = num_examples // batch_size
    #     print('epoch', epoch_size)
    example_indices = list(range(num_examples))

    #     print(example_indices)
    #     random.shuffle(example_indices)

    def _data(pos, data):
        if data is X:
            #             print(pos, pos+num_steps)
            #             print(data[pos:pos + num_steps, :, :, :].size())
            return data[pos:pos + num_steps, :, :, :]
        if data is Y:
            #             print(pos)
            return data[pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        #         print(batch_indices)
        XX = [_data(j, X) for j in batch_indices]
        YY = [_data(j, Y) for j in batch_indices]
        XX = torch.stack(XX)
        YY = torch.stack(YY)
        XX = XX.transpose(1, 2)
        yield XX, YY


lr, num_epochs = 0.001, 500
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
batch_size = 128
device = 'cpu'
train_ch5(net, batch_size, optimizer, device, num_epochs)
