from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # numpy -> Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 ，范围[-1,1]
])
train = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
test = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
