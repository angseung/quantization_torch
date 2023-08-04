"""
it overrides torchvision.models.segmentation.fcn
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import copy
import time
from functools import partial
from typing import Any, Optional, Dict

import torch
from torch import Tensor
from torch import nn
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

from quantization_torch.models.resnet import (
    QuantizableResNet,
    resnet101,
    ResNet101_Weights,
    resnet50,
    ResNet50_Weights,
    fuse_resnet,
)
from quantization_torch.utils.quantization_utils import get_platform_aware_qconfig
from quantization_torch.utils.segmentation_utils import _QuantizableSimpleSegmentationModel


__all__ = [
    "QuantizableFCN",
    "FCN_ResNet50_Weights",
    "FCN_ResNet101_Weights",
    "fcn_resnet50",
    "fcn_resnet101",
]


class QuantizableFCN(_QuantizableSimpleSegmentationModel):
    """
    Implements FCN model from
    `"Fully Convolutional Networks for Semantic Segmentation"
    <https://arxiv.org/abs/1411.4038>`_.

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
        fuse_fcn(self, is_qat=is_qat)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.quant(x)
        x = self._forward_impl(x)
        result = {k: self.dequant(v) for k, v in x.items()}

        return result


class QuantizableFCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class FCN_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 35322218,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 60.5,
                    "pixel_acc": 91.4,
                }
            },
            "_ops": 152.717,
            "_file_size": 135.009,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class FCN_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 54314346,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet101",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 63.7,
                    "pixel_acc": 91.9,
                }
            },
            "_ops": 232.738,
            "_file_size": 207.711,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def _fcn_resnet(
    backbone: QuantizableResNet,
    num_classes: int,
    aux: Optional[bool],
) -> QuantizableFCN:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = QuantizableFCNHead(1024, num_classes) if aux else None
    classifier = QuantizableFCNHead(2048, num_classes)
    return QuantizableFCN(backbone, classifier, aux_classifier)


@handle_legacy_interface(
    weights=("pretrained", FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def fcn_resnet50(
    *,
    weights: Optional[FCN_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> QuantizableFCN:
    """Fully-Convolutional Network model with a ResNet-50 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained
            weights for the backbone.
        quantize (bool): If True, returned model is prepared for PTQ or QAT
        is_qat (bool): If quantize and is_qat are both True, returned model is prepared for QAT
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet50_Weights
        :members:
    """
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    weights = FCN_ResNet50_Weights.verify(weights)
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
        weights=weights_backbone,
        replace_stride_with_dilation=[False, True, True],
        quantize=quantize,
        is_qat=is_qat,
        skip_fuse=True,
    )
    model = _fcn_resnet(backbone, num_classes, aux_loss)

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


@handle_legacy_interface(
    weights=("pretrained", FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet101_Weights.IMAGENET1K_V1),
)
def fcn_resnet101(
    *,
    weights: Optional[FCN_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V1,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> QuantizableFCN:
    """Fully-Convolutional Network model with a ResNet-101 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained
            weights for the backbone.
        quantize (bool): If True, returned model is prepared for PTQ or QAT
        is_qat (bool): If quantize and is_qat are both True, returned model is prepared for QAT
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet101_Weights
        :members:
    """
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    weights = FCN_ResNet101_Weights.verify(weights)
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
        weights=weights_backbone,
        replace_stride_with_dilation=[False, True, True],
        quantize=quantize,
        is_qat=is_qat,
        skip_fuse=True,
    )
    model = _fcn_resnet(backbone, num_classes, aux_loss)

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


def fuse_fcn(model: nn.Module, is_qat: bool = False) -> None:
    fuse_resnet(model.backbone, is_qat=is_qat)
    _fuse_modules(
        model.classifier,
        modules_to_fuse=[
            ["0", "1", "2"],
        ],
        is_qat=is_qat,
        inplace=True,
    )
    _fuse_modules(
        model.aux_classifier,
        modules_to_fuse=[
            ["0", "1", "2"],
        ],
        is_qat=is_qat,
        inplace=True,
    )


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = fcn_resnet101(
        weights=FCN_ResNet101_Weights.DEFAULT, quantize=True, is_qat=False
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
    #     f="../onnx/fcn_resnet50_fp.onnx",
    #     opset_version=13,
    # )  # success
    # torch.onnx.export(
    #     model,
    #     x,
    #     f="../onnx/fcn_resnet50_qint8.onnx",
    #     opset_version=13,
    # )  # failed
