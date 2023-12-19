import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models


def build_model(model_name):
    if model_name == "resnet18":
        from models.resnet import resnet18
        model = resnet18()
    elif model_name == "resnet50":
        from models.resnet import resnet50
        model = resnet50()
    elif model_name == "resnet101":
        from models.resnet import resnet101
        model = resnet101()

    ckpt_dir = f"ckpt/{model_name}.pth"

    cudnn.benchmark = True
    state_dic = torch.load(ckpt_dir)
    model.load_state_dict(state_dic)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model
