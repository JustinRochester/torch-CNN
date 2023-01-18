from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from LeNet import LeNet5
from torch import optim
from tqdm import trange

data_path = 'data'
train = datasets.MNIST(data_path, download=True, train=True)
test = datasets.MNIST(data_path, download=True, train=False)

X_train = train.data.unsqueeze(1) / 255.0
y_train = train.targets
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

X_test = test.data.unsqueeze(1) / 255.0
y_test = test.targets

model = LeNet5()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in trange(10):
    for X, y in train_loader:
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train)
        acc_train = (y_pred.argmax(dim=1) == y_train).float().mean().item()
        y_pred = model(X_test)
        acc_test = (y_pred.argmax(dim=1) == y_test).float().mean().item()
        print(epoch, acc_train, acc_test)
