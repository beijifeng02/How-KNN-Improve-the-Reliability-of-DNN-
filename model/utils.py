import torch
import torch.backends.cudnn as cudnn


def build_model(model_name):
    ckpt_path = f"ckpt/{model_name}.pth"
    if model_name == "resnet18":
        from .models.resnet import ResNet18
        model = ResNet18()
    elif model_name == "resnet50":
        from .models.resnet import ResNet50
        model = ResNet50()
    elif model_name == "resnet101":
        from .models.resnet import ResNet101
        model = ResNet101()
    else:
        raise ValueError("This models is not supported.")

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    state_dic = torch.load(ckpt_path)
    model.load_state_dict(state_dic["net"])
    model.eval()
    return model
