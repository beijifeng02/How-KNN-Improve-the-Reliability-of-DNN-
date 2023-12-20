import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models


def build_model(cfg):
    dataset = cfg.DATA.NAME
    model_name = cfg.MODEL.ARCH
    ckpt_dir = cfg.MODEL.CKPT_DIR

    if dataset == "cifar10" or dataset == "svhn":
        if model_name == "resnet18":
            from .cifar10.resnet import ResNet18
            model = ResNet18()
        elif model_name == "resnet50":
            from .cifar10.resnet import ResNet50
            model = ResNet50()
        elif model_name == "resnet101":
            from .cifar10.resnet import ResNet101
            model = ResNet101()
        else:
            raise ValueError("This models is not supported.")

        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        state_dic = torch.load(ckpt_dir)
        model.load_state_dict(state_dic["net"])

    elif dataset == "cifar100":
        if model_name == "resnet18":
            from .cifar100.resnet import resnet18
            model = resnet18()
        elif model_name == "resnet50":
            from .cifar100.resnet import resnet50
            model = resnet50()
        elif model_name == "resnet101":
            from .cifar100.resnet import resnet101
            model = resnet101()
        else:
            raise ValueError("This models is not supported.")

        cudnn.benchmark = True
        state_dic = torch.load(ckpt_dir)
        model.load_state_dict(state_dic)
        model = torch.nn.DataParallel(model)

    else:
        raise NotImplementedError(f"dataset {dataset} is not supported.")

    model.eval()
    return model
