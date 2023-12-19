import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ----------------------------- transform -------------------------------- #
# cifar100 transform
transform_cifar100_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

transform_cifar100_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])


# ----------------------------------------------------------------------- #
def build_dataloader(data_dir="/data/dataset/", batch_size=258, num_workers=8):
    trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_cifar100_train)
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_cifar100_test)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)
    return trainloader, testloader
