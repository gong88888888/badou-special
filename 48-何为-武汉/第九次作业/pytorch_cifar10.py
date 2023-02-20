import torch
from torch import nn
import torchvision

epoch = 100
batch_size = 64
lr = 0.01


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(  # 3*32*32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(  # 32*16*16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # 64*8*8
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Sequential(nn.Flatten(),
                                 nn.Linear(128 * 4 * 4, 1024), nn.ReLU(),
                                 nn.Linear(1024, 512), nn.ReLU(),
                                 nn.Linear(512, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out_put = self.out(x)
        return out_put


def train_model(net, train_data, optim, loss_func):
    for i in range(epoch):
        print('--- Epoch %d ---' % (i + 1))
        steps = 0
        total_loss = 0
        total_num = 0
        acc_num = 0
        net.train()
        for x, label in train_data:
            y = net(x)
            loss = loss_func(y, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.data
            pred = torch.argmax(nn.Softmax(dim=1)(y), dim=1)
            acc_num += (pred == label).sum()
            total_num += y.shape[0]
            steps += 1

        print('epoch: {}, loss: {}, acc: {}'.format(i + 1, total_loss / steps, acc_num / total_num))


def predict(net, test_data):
    net.eval()
    acc_num = 0
    total_num = 0
    net.eval()
    for x, label in test_data:
        y = net(x)
        pred = torch.argmax(nn.Softmax(dim=1)(y), dim=1)
        acc_num += (pred == label).sum()
        total_num += y.shape[0]

    print('train data acc: {}'.format(acc_num / total_num))


def get_dataset():
    CIFAR_train = torchvision.datasets.CIFAR10(root='./DataSet', train=True, download=True,
                                               transform=torchvision.transforms.ToTensor())

    CIFAR_test = torchvision.datasets.CIFAR10(root='./DataSet', train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())

    MNIST_train_data = torch.utils.data.DataLoader(CIFAR_train, batch_size=batch_size, shuffle=True)
    MNIST_test_data = torch.utils.data.DataLoader(CIFAR_test, batch_size=batch_size, shuffle=False)
    return MNIST_train_data, MNIST_test_data


if __name__ == '__main__':
    my_model = MyNet()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_data, test_data = get_dataset()

    train_model(my_model, train_data, optimizer, loss_func)
    predict(my_model, test_data)
    # print(len(MNIST_train))
    # print(len(MNIST_test))
