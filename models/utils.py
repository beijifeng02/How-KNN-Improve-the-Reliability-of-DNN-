import torch
import torch.backends.cudnn as cudnn
import torchvision


def build_model(cfg):
    dataset = cfg.DATA.NAME
    model_name = cfg.MODEL.ARCH
    ckpt_dir = cfg.MODEL.CKPT_DIR

    if dataset == "cifar10" or dataset == "svhn":
        if model_name == "resnet18":
            from .resnet import resnet18
            model = resnet18()
        elif model_name == "resnet50":
            from .resnet import resnet50
            model = resnet50()
        elif model_name == "resnet101":
            from .resnet import resnet101
            model = resnet101()
        else:
            raise ValueError("This models is not supported.")

        checkpoint = torch.load(ckpt_dir, map_location='cpu')
        checkpoint = {'net': {key.replace("module.", ""): value for key, value in checkpoint['net'].items()}}
        model.load_state_dict(checkpoint['net'])

    elif dataset == "cifar100":
        if model_name == "resnet18":
            from .resnet import resnet18
            model = resnet18(num_classes=100)
        elif model_name == "resnet50":
            from .resnet import resnet50
            model = resnet50(num_classes=100)
        elif model_name == "resnet101":
            from .resnet import resnet101
            model = resnet101(num_classes=100)
        else:
            raise ValueError("This models is not supported.")

    else:
        raise NotImplementedError(f"dataset {dataset} is not supported.")

    checkpoint = torch.load(ckpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    return model
