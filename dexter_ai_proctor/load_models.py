import torch

from .models.common import DetectMultiBackend
from .utils import stable_hopenetlite
from .utils.general import check_img_size
from .utils.model_celebmask import BiSeNet
from .utils.torch_utils import select_device

def load_yolov5(model_path, imgsz=640, half=False):
    """funtion to load yolov5 model

    Args:
        model_path (str): path to pt file
        imgsz (int, optional): size of image to run inference on. Defaults to 640.
        half (bool, optional): whether to run in fp16 or fp32 mode. Defaults to False.

    Returns:
        model, device, stride, names, pt, imgsz, half: required for inference
    """
    device = select_device('')
    model = DetectMultiBackend(model_path, device=device, dnn=False, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
    return model, device, stride, names, pt, imgsz, half

def load_face_segmentation(model_path):
    """function to load face segmentation model

    Args:
        model_path (str): path to pth file

    Returns:
        model, device: required for inference
    """
    model = BiSeNet(n_classes=19)
    device = select_device('')
    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, device

def load_head_pose(model_path):
    """function to load head pose model

    Args:
        model_path (str): path to pkl file

    Returns:
        model, device, idx_tensor: required for inference
    """
    model = stable_hopenetlite.shufflenet_v2_x1_0()
    device = select_device('')
    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    return model, device, idx_tensor
