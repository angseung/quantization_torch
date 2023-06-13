import os.path
import platform
from pathlib import Path
import cv2
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.general import non_max_suppression
from utils.torch_utils import normalizer
from utils.roi_utils import resize
from utils.augmentations import wrap_letterbox
from models.yolo import YoloBackboneQuantizer, YoloHead

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # root directory


class QuantizableModel(nn.Module):
    def __init__(self, model: nn.Module, is_qat: bool = False):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        self.model = model.to("cpu")

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.is_qat = is_qat
        self._set_qconfig()

    def _set_qconfig(self):
        arch = platform.machine()

        if "AMD64" in arch or "x86_64" in arch:
            self.arch = "x86"
        elif "aarch64" in arch or "arm64" in arch:
            self.arch = "qnnpack"

        self.qconfig = (
            torch.ao.quantization.get_default_qat_qconfig(self.arch)
            if self.is_qat
            else torch.ao.quantization.get_default_qconfig(self.arch)
        )

    def prepare(self) -> nn.Module:
        if self.is_qat:
            torch.ao.quantization.prepare_qat(self.train(), inplace=True)
        else:
            torch.ao.quantization.prepare(self.eval(), inplace=True)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x


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
    # fname = os.path.join("weights", "yolov5l-qat.pt")
    fname = os.path.join("models", "yolov4-qat.yaml")
    yolo_detector = YoloHead(fname)
    yolo_fp32 = YoloBackboneQuantizer(fname, yolo_version=4)
    yolo_qint8 = YoloBackboneQuantizer(fname, yolo_version=4)
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
