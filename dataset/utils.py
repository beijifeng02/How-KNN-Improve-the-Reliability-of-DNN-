import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ----------------------------- transform -------------------------------- #
# cifar10 transform
transform_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# ----------------------------------------------------------------------- #
def build_dataloader(data_dir="/data/dataset/", batch_size=258, num_workers=8):
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_cifar10_test)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)
    return testloader
