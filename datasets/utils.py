import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ----------------------------- transform -------------------------------- #
# imagenet transform
transform_imagenet_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_imagenet_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# ----------------------------------------------------------------------- #
def build_dataloader(data_dir="/data/dataset/", batch_size=258, num_workers=8):
    traindir = os.path.join(data_dir, 'imagenet/images/train')
    validir = os.path.join(data_dir, 'imagenet/images/val')
    trainset = datasets.ImageFolder(root=traindir, transform=transform_imagenet_train)
    testset = datasets.ImageFolder(root=validir, transform=transform_imagenet_test)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)
    return trainloader, testloader
