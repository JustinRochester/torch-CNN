from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from VGG16Net import VGGNet
from torch import optim
from tqdm import trange

data_path = 'data/CIFAR10'
train = datasets.CIFAR10(data_path, download=True, train=True)
test = datasets.CIFAR10(data_path, download=True, train=False)

batch_size = 2048
X_train = torch.Tensor(train.data.transpose(0, 3, 1, 2)).cuda()
y_train = torch.Tensor(train.targets).long().cuda()
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

X_test = torch.Tensor(test.data.transpose(0, 3, 1, 2)).cuda()
y_test = torch.Tensor(test.targets).long().cuda()
train_evaluate_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size * 2, shuffle=False)
test_evaluate_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size * 2, shuffle=False)

model = VGGNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

n = y_train.shape[0]
epoch_number = 100
max_test_acc_value = 0
max_test_acc_turn = -1
for epoch in trange(epoch_number):
    count = 0
    for X, y in train_loader:
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = min(count + batch_size, n)
        print('finished epoch {} / {} ,batches {} / {}, loss = {}'.format(epoch, epoch_number, count, n, loss.item()))
    with torch.no_grad():
        acc_train = 0
        for X, y in train_evaluate_loader:
            pred = model(X)
            acc_train += (pred.argmax(dim=1) == y).float().sum().item()
        acc_test = 0
        for X, y in train_evaluate_loader:
            pred = model(X)
            acc_test += (pred.argmax(dim=1) == y).float().sum().item()
        acc_train /= n
        acc_test /= n
        acc_train *= 100
        acc_test *= 100
        print('finished epoch {} / {}, train accuracy = {}%, test accuracy = {}%'.format(epoch+1, epoch_number, acc_train, acc_test))
        if acc_test > max_test_acc_value:
            max_test_acc_value = acc_test
            max_test_acc_turn = epoch + 1
print('finished training, got max test accuracy = {}% in turn {} / {}'.format(max_test_acc_value, max_test_acc_turn, epoch_number))
