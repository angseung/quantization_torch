import os
import platform
from pathlib import Path
import cv2
import torch
from torch import Tensor
from torch import nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss
from torchvision import transforms
from quantization_torch.utils.torch_utils import normalizer
from quantization_torch.utils.roi_utils import resize
from quantization_torch.utils.augmentations import wrap_letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # root directory


class QuantizableModel(nn.Module):
    """
    Deprecated
    Wrapper Class for Quantizable Model
    """

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
            torch.backends.quantized.engine = self.arch

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

    def forward(self, x: Tensor) -> Tensor:
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


def get_platform_aware_qconfig() -> str:
    arch = platform.machine()

    if "AMD64" in arch or "x86_64" in arch:
        arch = "x86"
    elif "aarch64" in arch or "arm64" in arch:
        arch = "qnnpack"

    else:
        raise RuntimeError(
            f"Quantization Not Available on this platform, {arch}. It supports only AArch64 and x86-64 system."
        )

    return arch


def cal_mse(pred: Tensor, target: Tensor, norm: bool = False) -> Tensor:
    if norm:
        mse = mse_loss(target, pred) / mse_loss(target, torch.zeros_like(target))

    else:
        mse = mse_loss(target, pred)

    return mse.cpu().detach()
