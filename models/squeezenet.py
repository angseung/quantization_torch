from functools import partial
from typing import Any, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.init as init

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.models.quantization.utils import _fuse_modules
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from utils.quantization_utils import cal_mse, get_platform_aware_qconfig


__all__ = [
    "QuantizableSqueezeNet",
    "SqueezeNet1_0_Weights",
    "SqueezeNet1_1_Weights",
    "squeezenet1_0",
    "squeezenet1_1",
]


class Fire(nn.Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )


class QuantizableSqueezeNet(nn.Module):
    def __init__(
        self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(
                f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected"
            )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def fuse_model(self, is_qat: bool):
        fuse_squeezenet(self, is_qat)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x


def _squeezenet(
    version: str,
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    is_qat: bool,
    **kwargs: Any,
) -> QuantizableSqueezeNet:
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = QuantizableSqueezeNet(version, **kwargs)
    model.eval()

    if quantize:
        if is_qat:
            model.fuse_model(is_qat=True)
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)
        else:
            model.fuse_model(is_qat=False)
            model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            torch.ao.quantization.prepare(model, inplace=True)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/pull/49#issuecomment-277560717",
    "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
}


class SqueezeNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "min_size": (21, 21),
            "num_params": 1248424,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 58.092,
                    "acc@5": 80.420,
                }
            },
            "_ops": 0.819,
            "_file_size": 4.778,
        },
    )
    DEFAULT = IMAGENET1K_V1


class SqueezeNet1_1_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "min_size": (17, 17),
            "num_params": 1235496,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 58.178,
                    "acc@5": 80.624,
                }
            },
            "_ops": 0.349,
            "_file_size": 4.729,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", SqueezeNet1_0_Weights.IMAGENET1K_V1))
def squeezenet1_0(
    *,
    weights: Optional[SqueezeNet1_0_Weights] = None,
    progress: bool = True,
    quantize=False,
    is_qat=False,
    **kwargs: Any,
) -> QuantizableSqueezeNet:
    """SqueezeNet model architecture from the `SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        weights (:class:`~torchvision.models.SqueezeNet1_0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.SqueezeNet1_0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.SqueezeNet1_0_Weights
        :members:
    """
    weights = SqueezeNet1_0_Weights.verify(weights)
    return _squeezenet("1_0", weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", SqueezeNet1_1_Weights.IMAGENET1K_V1))
def squeezenet1_1(
    *,
    weights: Optional[SqueezeNet1_1_Weights] = None,
    progress: bool = True,
    quantize: bool,
    is_qat: bool,
    **kwargs: Any,
) -> QuantizableSqueezeNet:
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.

    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        weights (:class:`~torchvision.models.SqueezeNet1_1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.SqueezeNet1_1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.SqueezeNet1_1_Weights
        :members:
    """
    weights = SqueezeNet1_1_Weights.verify(weights)
    return _squeezenet("1_1", weights, progress, quantize, is_qat, **kwargs)


def fuse_squeezenet(model: nn.Module, is_qat: bool = False) -> None:
    for module_name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            if module_name == "features":
                _fuse_modules(
                    module,
                    [["0", "1"]],  # Conv-ReLU
                    is_qat=is_qat,
                    inplace=True,
                )

                for _, block in module.named_children():
                    if isinstance(block, Fire):
                        _fuse_modules(
                            block,
                            [
                                ["squeeze", "squeeze_activation"],  # Conv-ReLU
                                ["expand1x1", "expand1x1_activation"],  # Conv-ReLU
                                ["expand3x3", "expand3x3_activation"],  # Conv-ReLU
                            ],
                            is_qat=is_qat,
                            inplace=True,
                        )

            elif module_name == "classifier":
                _fuse_modules(
                    module,
                    [["1", "2"]],  # Conv-ReLU
                    is_qat=is_qat,
                    inplace=True,
                )


if __name__ == "__main__":
    # model = squeezenet1_0(quantize=True, is_qat=False)
    model = squeezenet1_1(quantize=True, is_qat=False)
    model_fp = copy.deepcopy(model)
    input = torch.randn(1, 3, 224, 224)
    model(input)
    torch.ao.quantization.convert(model, inplace=True)
    dummy_output = model(input)
    dummy_output_fp = model_fp(input)
    mse = cal_mse(dummy_output, dummy_output_fp, norm=True)

    from utils.onnx_utils import convert_onnx

    convert_onnx(model, "../onnx/squeezenet1_1_qint8.onnx", opset=13)
    convert_onnx(model_fp, "../onnx/squeezenet1_1_fp32.onnx", opset=13)
