"""
it overrides torchvision.models.quantization.mobilenetv2
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import copy
from functools import partial
from typing import Any, Optional, Union

import torch
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.models.mobilenetv2 import (
    InvertedResidual,
    MobileNet_V2_Weights,
    MobileNetV2,
)
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.transforms._presets import ImageClassification
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.quantization.utils import (
    _fuse_modules,
    _replace_relu,
)

from quantization_torch.utils.quantization_utils import get_platform_aware_qconfig, cal_mse


__all__ = [
    "QuantizableMobileNetV2",
    "MobileNet_V2_QuantizedWeights",
    "mobilenet_v2",
]


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self, is_qat: bool = False) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                _fuse_modules(self.conv, [str(idx), str(idx + 1)], is_qat, inplace=True)


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: bool = False) -> None:
        fuse_mobilenetv2(self, is_qat=is_qat)


class MobileNet_V2_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 3504872,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "backend": "qnnpack",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv2",
            "unquantized": MobileNet_V2_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.658,
                    "acc@5": 90.150,
                }
            },
            "_ops": 0.301,
            "_file_size": 3.423,
            "_docs": """
                These weights were produced by doing Quantization Aware Training (eager mode) on top of the unquantized
                weights listed below.
            """,
        },
    )
    DEFAULT = IMAGENET1K_QNNPACK_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V2_Weights.IMAGENET1K_V1,
    )
)
def mobilenet_v2(
    *,
    weights: Optional[
        Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]
    ] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    skip_fuse: Optional[bool] = False,
    **kwargs: Any,
) -> QuantizableMobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <https://arxiv.org/abs/1801.04381>`_.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` or :class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        quantize (bool, optional): If True, returns a quantized version of the model. Default is False.
        is_qat:
        skip_fuse
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableMobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv2.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.quantization.MobileNet_V2_QuantizedWeights
        :members:
    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
        :noindex:
    """
    weights = (
        MobileNet_V2_QuantizedWeights if quantize else MobileNet_V2_Weights
    ).verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])

    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    _replace_relu(model)

    model.eval()

    if not skip_fuse:
        model.fuse_model(is_qat=is_qat)

    if quantize:
        if is_qat:
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)

        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            torch.ao.quantization.prepare(model, inplace=True)

    return model


def fuse_mobilenetv2(model: nn.Module, is_qat: bool = False) -> None:
    for m in model.modules():
        if type(m) is Conv2dNormActivation:
            _fuse_modules(m, ["0", "1", "2"], is_qat=is_qat, inplace=True)
        if type(m) is QuantizableInvertedResidual:
            m.fuse_model(is_qat)


if __name__ == "__main__":
    model = mobilenet_v2(quantize=True, is_qat=False)
    model_fp = copy.deepcopy(model)
    input = torch.randn(1, 3, 224, 224)
    model(input)  # Calibration codes here...
    torch.ao.quantization.convert(model, inplace=True)
    dummy_output = model(input)
    dummy_output_fp = model_fp(input)
    nmse = cal_mse(dummy_output, dummy_output_fp, norm=False)
