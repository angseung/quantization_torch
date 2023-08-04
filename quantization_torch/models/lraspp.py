"""
it overrides torchvision.models.segmentation.lraspp
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import copy
import time
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.ao.quantization import DeQuantStub, QuantStub
from torch.ao.nn.quantized import FloatFunctional
from torchvision.models.quantization.utils import _fuse_modules
from torchvision.transforms._presets import SemanticSegmentation
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _VOC_CATEGORIES
from torchvision.models._utils import (
    _ovewrite_value_param,
    handle_legacy_interface,
    IntermediateLayerGetter,
)

from quantization_torch.models.mobilenetv3 import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    QuantizableMobileNetV3,
    fuse_mobilenetv3,
)
from quantization_torch.utils.quantization_utils import get_platform_aware_qconfig


__all__ = [
    "QuantizableLRASPP",
    "LRASPP_MobileNet_V3_Large_Weights",
    "lraspp_mobilenet_v3_large",
]


class QuantizableLRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int, optional): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = QuantizableLRASPPHead(
            low_channels, high_channels, num_classes, inter_channels
        )
        self.quant = QuantStub()

    def fuse_model(self, is_qat: bool = False) -> None:
        fuse_lraspp(self, is_qat=is_qat)

    def _forward_impl(self, input: Tensor) -> Dict[str, Tensor]:
        input = self.quant(input)
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(
            out, size=input.shape[-2:], mode="bilinear", align_corners=False
        )

        result = OrderedDict()
        result["out"] = out

        return result

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        result = self._forward_impl(input)

        return result


class QuantizableLRASPPHead(nn.Module):
    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int,
    ) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)
        self.skip_add = FloatFunctional()
        self.skip_mul = FloatFunctional()
        self.dequant = DeQuantStub()

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = self.skip_mul.mul(x, s)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.skip_add.add(self.low_classifier(low), self.high_classifier(x))

        return self.dequant(x)


def _lraspp_mobilenetv3(
    backbone: QuantizableMobileNetV3, num_classes: int
) -> QuantizableLRASPP:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)]
        + [len(backbone) - 1]
    )
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(
        backbone, return_layers={str(low_pos): "low", str(high_pos): "high"}
    )

    return QuantizableLRASPP(backbone, low_channels, high_channels, num_classes)


class LRASPP_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            "num_params": 3221538,
            "categories": _VOC_CATEGORIES,
            "min_size": (1, 1),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#lraspp_mobilenet_v3_large",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 57.9,
                    "pixel_acc": 91.2,
                }
            },
            "_ops": 2.086,
            "_file_size": 12.49,
            "_docs": """
                These weights were trained on a subset of COCO, using only the 20 categories that are present in the
                Pascal VOC dataset.
            """,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


@handle_legacy_interface(
    weights=("pretrained", LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def lraspp_mobilenet_v3_large(
    *,
    weights: Optional[LRASPP_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[
        MobileNet_V3_Large_Weights
    ] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> QuantizableLRASPP:
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        quantize (bool): If True, returned model is prepared for PTQ or QAT
        is_qat (bool): If quantize and is_qat are both True, returned model is prepared for QAT
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights
        :members:
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    weights = LRASPP_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(
            "num_classes", num_classes, len(weights.meta["categories"])
        )
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(
        weights=weights_backbone,
        dilated=True,
        quantize=quantize,
        is_qat=is_qat,
        skip_fuse=True,
    )
    model = _lraspp_mobilenetv3(backbone, num_classes)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)

    model.eval()
    model.fuse_model(is_qat=is_qat)

    if quantize:
        if is_qat:
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
            model.backbone.qconfig = torch.ao.quantization.get_default_qat_qconfig(
                backend
            )
            model.classifier.qconfig = torch.ao.quantization.get_default_qat_qconfig(
                backend
            )
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)

        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.backbone.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.classifier.qconfig = torch.ao.quantization.get_default_qconfig(
                backend
            )
            torch.ao.quantization.prepare(model, inplace=True)

    return model


def fuse_lraspp(model: nn.Module, is_qat: bool = False) -> None:
    fuse_mobilenetv3(model.backbone, is_qat=is_qat)
    _fuse_modules(model.classifier.cbr, [["0", "1", "2"]], is_qat=is_qat, inplace=True)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = lraspp_mobilenet_v3_large(
        weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        quantize=True,
        is_qat=False,
    )
    model.eval()
    model_fp = copy.deepcopy(model)
    model(x)
    torch.ao.quantization.convert(model, inplace=True)

    start = time.time()
    predictions = model(x)
    elapsed_quant = time.time() - start

    start = time.time()
    predictions_fp = model_fp(x)
    elapsed_fp = time.time() - start

    print(f"latency_quant: {elapsed_quant: .2f}, latency_fp: {elapsed_fp: .2f}")

    # torch.onnx.export(
    #     model_fp,
    #     x,
    #     f="../onnx/lraspp_mobilenetv3_fp.onnx",
    #     opset_version=13,
    # )  # success
    # torch.onnx.export(
    #     model,
    #     x,
    #     f="../onnx/lraspp_mobilenetv3_qint8.onnx",
    #     opset_version=13,
    # )  # failed
