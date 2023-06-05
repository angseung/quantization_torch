import os.path
import platform
import copy
from typing import *
from pathlib import Path
import cv2
import torch
from torch import nn as nn
from torch.ao.quantization import fuse_modules
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.common import ConvBnReLU, Concat, DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import normalizer
from utils.roi_utils import resize
from utils.augmentations import wrap_letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # root directory


class QuantizableModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        self.model = model.to("cpu")

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x


class YoloBackboneQuantizer(nn.Module):
    def __init__(self, model: Union[str, nn.Module] = None, yolo_version: int = 3):
        super().__init__()
        self.yolo_version = yolo_version
        self.is_fused = False

        if isinstance(model, str):
            if model.endswith(".pt"):
                self.model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
                if isinstance(self.model, dict):
                    self.model = self.model["model"].float()
            elif model.endswith(".yaml"):
                self.model = DetectMultiBackend(
                    os.path.join(ROOT, model), torch.device("cpu"), False
                )
        elif isinstance(model, nn.Module):
            self.model = model

        elif isinstance(model, DetectMultiBackend):
            self.model = model.model

        else:
            raise AttributeError("Unsupported model type")

        self.model.eval()
        self.quant = torch.ao.quantization.QuantStub()
        self.model.model[-1] = nn.Identity()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def fuse_model(self):
        fuse_conv_bn_relu(self.model)
        self.is_fused = True

    def _forward_impl_v3(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.quant(x)

        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "18":
                    x = block([x17, x8])
                elif i == "25":
                    x = block([x24, x6])
            else:  # ConvBnReLU, BottleneckReLU
                if i == "16":
                    x = block(x14)
                elif i == "23":
                    x = block(x21)
                else:
                    x = block(x)

                # save feature map for concat/conv layers...
                if i == "6":
                    x6 = x.clone()
                elif i == "8":
                    x8 = x.clone()
                elif i == "14":
                    x14 = x.clone()
                elif i == "15":
                    x15 = x.clone()
                elif i == "17":
                    x17 = x.clone()
                elif i == "21":
                    x21 = x.clone()
                elif i == "22":
                    x22 = x.clone()
                elif i == "24":
                    x24 = x.clone()
                elif i == "27":
                    x27 = x.clone()

        x15 = self.dequant(x15)
        x22 = self.dequant(x22)
        x27 = self.dequant(x27)

        return [x27, x22, x15]

    def _forward_impl_v4(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.quant(x)

        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "25":
                    x = block([x24, x12])
                elif i == "29":
                    x = block([x28, x9])
                elif i == "32":
                    x = block([x31, x27])
                elif i == "35":
                    x = block([x34, x23])
            else:  # ConvBnReLU, BottleneckReLU, BottleneckCSPReLU, SPPReLU
                x = block(x)

                # save feature map for concat/conv layers...
                if i == "9":
                    x9 = x.clone()
                elif i == "12":
                    x12 = x.clone()
                elif i == "23":
                    x23 = x.clone()
                elif i == "24":
                    x24 = x.clone()
                elif i == "27":
                    x27 = x.clone()
                elif i == "28":
                    x28 = x.clone()
                elif i == "30":
                    x30 = x.clone()
                elif i == "31":
                    x31 = x.clone()
                elif i == "33":
                    x33 = x.clone()
                elif i == "34":
                    x34 = x.clone()
                elif i == "36":
                    x36 = x.clone()

        x30 = self.dequant(x30)
        x33 = self.dequant(x33)
        x36 = self.dequant(x36)

        return [x30, x33, x36]

    def _forward_impl_v5(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.quant(x)

        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "12":
                    x = block([x11, x6])
                elif i == "16":
                    x = block([x15, x4])
                elif i == "19":
                    x = block([x18, x14])
                elif i == "22":
                    x = block([x21, x10])
            else:  # ConvBnReLU, BottleneckReLU, C3ReLU, SPPFReLU
                x = block(x)

                # save feature map for concat/conv layers...
                if i == "4":
                    x4 = x.clone()
                elif i == "6":
                    x6 = x.clone()
                elif i == "10":
                    x10 = x.clone()
                elif i == "11":
                    x11 = x.clone()
                elif i == "14":
                    x14 = x.clone()
                elif i == "15":
                    x15 = x.clone()
                elif i == "17":
                    x17 = x.clone()
                elif i == "18":
                    x18 = x.clone()
                elif i == "20":
                    x20 = x.clone()
                elif i == "21":
                    x21 = x.clone()
                elif i == "23":
                    x23 = x.clone()

        x17 = self.dequant(x17)
        x20 = self.dequant(x20)
        x23 = self.dequant(x23)

        return [x17, x20, x23]

    def forward(
        self, x: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        if self.yolo_version == 3:
            return self._forward_impl_v3(x)

        elif self.yolo_version == 4:
            return self._forward_impl_v4(x)

        elif self.yolo_version == 5:
            return self._forward_impl_v5(x)


class YoloHead(nn.Module):
    def __init__(self, model: Union[str, nn.Module] = None):
        super().__init__()
        if isinstance(model, str):
            if model.endswith(".pt"):
                yolo_model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
                if isinstance(yolo_model, dict):
                    yolo_model = yolo_model["model"].float()
            elif model.endswith(".yaml"):
                yolo_model = DetectMultiBackend(
                    os.path.join(model), torch.device("cpu"), False
                )
        elif isinstance(model, nn.Module):
            yolo_model = model

        elif isinstance(model, DetectMultiBackend):
            yolo_model = model.model

        else:
            raise AttributeError("Unsupported model type")

        self.model = copy.deepcopy(yolo_model.model[-1])
        self.model.eval()

    def forward(
        self, x: List[torch.Tensor]
    ) -> Union[Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        return self.model(x) if self.model.training else self.model(x)[0]


def fuse_conv_bn_relu(blocks: nn.Module):
    """
    A function for fusing conv-bn-relu layers
    Parameters
    ----------
    blocks: A nn.Module type model to be fused
    -------

    """
    for _, block in blocks.named_children():
        if isinstance(block, ConvBnReLU):
            fuse_modules(block, [["conv", "bn", "act"]], inplace=True)
        else:
            fuse_conv_bn_relu(block)


class CalibrationDataLoader(Dataset):
    def __init__(self, img_dir: str, target_size: int = 320):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(), normalizer()])
        self.target_size = target_size
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)

    def __getitem__(self, item):
        img = cv2.imread(f"{self.img_dir}/{self.img_list[item]}")  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        img = resize(img, self.target_size)
        img = wrap_letterbox(img, self.target_size)[0]  # padded as square shape

        return self.transform(img)

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    # create a model instance
    architecture = platform.uname().machine
    dataset = CalibrationDataLoader(os.path.join(ROOT, "data", "cropped"))
    calibration_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input = torch.randn(1, 3, 320, 320)
    fname = os.path.join("weights", "yolov3-nano-qat.pt")
    yolo_detector = YoloHead(fname)
    yolo_fp32 = YoloBackboneQuantizer(fname, yolo_version=3)
    yolo_qint8 = YoloBackboneQuantizer(fname, yolo_version=3)
    yolo_qint8.fuse_model()

    if "AMD64" in platform.machine() or "x86_64" in platform.machine():
        yolo_qint8.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    elif "aarch64" in platform.machine() or "arm64" in platform.machine():
        torch.backends.quantized.engine = "qnnpack"
        yolo_qint8.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")

    torch.ao.quantization.prepare(yolo_qint8, inplace=True)

    for i, img in enumerate(calibration_dataloader):
        print(f"\rcalibrating... {i + 1} / {dataset.__len__()}", end="")
        yolo_qint8(img)

    torch.ao.quantization.convert(yolo_qint8, inplace=True)
    dummy_output = yolo_qint8(input)
    pred = yolo_detector(dummy_output)

    pred_fp32 = yolo_detector(yolo_fp32(input))

    pred_qint = non_max_suppression(
        pred,
        0.1,
        0.25,
    )

    # onnx export test
    torch.onnx.export(
        yolo_qint8,
        input,
        "../yolov3_backbone_qint8.onnx",
        opset_version=13,
    )
