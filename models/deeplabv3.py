"""
it overrides torchvision.models.segmentation.deeplabv3
"""

import copy
import time
from functools import partial
from typing import Any, List, Optional, Dict
from collections import OrderedDict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.models.quantization.utils import _fuse_modules
from torchvision.transforms._presets import SemanticSegmentation
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _VOC_CATEGORIES
from torchvision.models._utils import (
    _ovewrite_value_param,
    handle_legacy_interface,
    IntermediateLayerGetter,
)

from models.mobilenetv3 import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    QuantizableMobileNetV3,
    fuse_mobilenetv3,
)
from models.resnet import (
    QuantizableResNet,
    resnet101,
    ResNet101_Weights,
    resnet50,
    ResNet50_Weights,
)
from models.fcn import QuantizableFCNHead
from utils.quantization_utils import get_platform_aware_qconfig
from utils.segmentation_utils import _QuantizableSimpleSegmentationModel


__all__ = [
    "DeepLabV3",
    "DeepLabV3_ResNet50_Weights",
    "DeepLabV3_ResNet101_Weights",
    "DeepLabV3_MobileNet_V3_Large_Weights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


class DeepLabV3(_QuantizableSimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def fuse_model(self, is_qat: bool = False) -> None:
        fuse_deeplabv3(self, is_qat=is_qat)

    def _forward_impl(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        x = self.quant(x)
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = self.dequant(x)

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = self.dequant(x)

        return result

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self._forward_impl(x)


class QuanttizableDeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, atrous_rates: List[int], out_channels: int = 256
    ) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(
    backbone: QuantizableResNet,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = QuantizableFCNHead(1024, num_classes) if aux else None
    classifier = QuanttizableDeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class DeepLabV3_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 42004074,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 66.4,
                    "pixel_acc": 92.4,
                }
            },
            "_ops": 178.722,
            "_file_size": 160.515,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 60996202,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet101",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 67.4,
                    "pixel_acc": 92.4,
                }
            },
            "_ops": 258.743,
            "_file_size": 233.217,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 11029328,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 60.3,
                    "pixel_acc": 91.2,
                }
            },
            "_ops": 10.452,
            "_file_size": 42.301,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def _deeplabv3_mobilenetv3(
    backbone: QuantizableMobileNetV3,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)]
        + [len(backbone) - 1]
    )
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        return_layers[str(aux_pos)] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = QuantizableFCNHead(aux_inplanes, num_classes) if aux else None
    classifier = QuanttizableDeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def deeplabv3_resnet50(
    *,
    weights: Optional[DeepLabV3_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for the
            backbone
        quantize (bool): If True, returned model is prepared for PTQ or QAT
        is_qat (bool): If quantize and is_qat are both True, returned model is prepared for QAT
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
        :members:
    """
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    weights = DeepLabV3_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(
            "num_classes", num_classes, len(weights.meta["categories"])
        )
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(
        weights=weights_backbone, replace_stride_with_dilation=[False, True, True]
    )
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

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
            model.backbone.classifier = torch.ao.quantization.get_default_qat_qconfig(
                backend
            )
            model.backbone.aux_classifier = (
                torch.ao.quantization.get_default_qat_qconfig(backend)
            )
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)

        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.backbone.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.classifier.qconfig = torch.ao.quantization.get_default_qconfig(
                backend
            )
            model.aux_classifier.qconfig = torch.ao.quantization.get_default_qconfig(
                backend
            )
            torch.ao.quantization.prepare(model, inplace=True)

    return model


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet101_Weights.IMAGENET1K_V1),
)
def deeplabv3_resnet101(
    *,
    weights: Optional[DeepLabV3_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V1,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        quantize (bool): If True, returned model is prepared for PTQ or QAT
        is_qat (bool): If quantize and is_qat are both True, returned model is prepared for QAT
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    weights = DeepLabV3_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(
            "num_classes", num_classes, len(weights.meta["categories"])
        )
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet101(
        weights=weights_backbone, replace_stride_with_dilation=[False, True, True]
    )
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

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
            model.backbone.classifier = torch.ao.quantization.get_default_qat_qconfig(
                backend
            )
            model.backbone.aux_classifier = (
                torch.ao.quantization.get_default_qat_qconfig(backend)
            )
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)

        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.backbone.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.classifier.qconfig = torch.ao.quantization.get_default_qconfig(
                backend
            )
            model.aux_classifier.qconfig = torch.ao.quantization.get_default_qconfig(
                backend
            )
            torch.ao.quantization.prepare(model, inplace=True)

    return model


@handle_legacy_interface(
    weights=(
        "pretrained",
        DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
    ),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def deeplabv3_mobilenet_v3_large(
    *,
    weights: Optional[DeepLabV3_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[
        MobileNet_V3_Large_Weights
    ] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained weights
            for the backbone
        quantize (bool): If True, returned model is prepared for PTQ or QAT
        is_qat (bool): If quantize and is_qat are both True, returned model is prepared for QAT
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights
        :members:
    """
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    weights = DeepLabV3_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(
            "num_classes", num_classes, len(weights.meta["categories"])
        )
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(
        weights=weights_backbone,
        dilated=True,
        quantize=quantize,
        is_qat=is_qat,
        skip_fuse=True,
    )
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

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
            model.backbone.classifier = torch.ao.quantization.get_default_qat_qconfig(
                backend
            )
            model.backbone.aux_classifier = (
                torch.ao.quantization.get_default_qat_qconfig(backend)
            )
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)

        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.backbone.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            model.classifier.qconfig = torch.ao.quantization.get_default_qconfig(
                backend
            )
            model.aux_classifier.qconfig = torch.ao.quantization.get_default_qconfig(
                backend
            )
            torch.ao.quantization.prepare(model, inplace=True)

    return model


def fuse_deeplabv3(model: nn.Module, is_qat: bool = False) -> None:
    fuse_mobilenetv3(model.backbone, is_qat=is_qat)
    _fuse_modules(model.aux_classifier, [["0", "1", "2"]], is_qat=is_qat, inplace=True)
    _fuse_modules(model.classifier, [["1", "2", "3"]], is_qat=is_qat, inplace=True)
    _fuse_modules(
        model.classifier[0].convs[0], [["0", "1", "2"]], is_qat=is_qat, inplace=True
    )
    _fuse_modules(
        model.classifier[0].project, [["0", "1", "2"]], is_qat=is_qat, inplace=True
    )
    fuse_deeplabv3_head(model.classifier, is_qat=is_qat)


def fuse_deeplabv3_head(model: nn.Module, is_qat: bool = False) -> None:
    for module_name, module in model.named_children():
        if isinstance(module, ASPPConv):
            _fuse_modules(module, [["0", "1", "2"]], is_qat=is_qat, inplace=True)
        elif isinstance(module, ASPPPooling):
            _fuse_modules(module, [["1", "2", "3"]], is_qat=is_qat, inplace=True)
        else:
            fuse_deeplabv3_head(module, is_qat=is_qat)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = deeplabv3_mobilenet_v3_large(
        weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        quantize=True,
        is_qat=False,
    )
    # model = deeplabv3_resnet50(
    #     weights=DeepLabV3_ResNet50_Weights.DEFAULT,
    #     weights_backbone=ResNet50_Weights.DEFAULT,
    #     quantize=True,
    #     is_qat=False,
    # )
    # model = deeplabv3_resnet101(
    #     weights=DeepLabV3_ResNet101_Weights.DEFAULT,
    #     weights_backbone=ResNet101_Weights.DEFAULT,
    #     quantize=True,
    #     is_qat=False,
    # )
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
    #     f="../onnx/deeplabv3_fp.onnx",
    #     opset_version=13,
    # )  # success
    # torch.onnx.export(
    #     model,
    #     x,
    #     f="../onnx/deeplabv3_qint8.onnx",
    #     opset_version=13,
    # )  # failed
