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
def build_dataloader(cfg):
    dataset = cfg.DATA.NAME
    data_dir = cfg.DATA.DATA_DIR
    batch_size = cfg.TEST.BATCH_SIZE
    num_workers = cfg.DATA.NUM_WORKERS
    calib_num = cfg.DATA.CALIB_NUM

    if dataset == "cifar10":
        trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_cifar10_train)
        testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_cifar10_test)

    elif dataset == "cifar100":
        trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_cifar100_train)
        testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_cifar100_test)

    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented!")

    calibset, testset = torch.utils.data.random_split(testset, [calib_num, 10000 - calib_num])
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers)
    calibloader = torch.utils.data.DataLoader(dataset=calibset, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)

    return trainloader, calibloader, testloader
