import copy
from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.models.quantization.utils import _fuse_modules
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

from utils.quantization_utils import cal_mse, get_platform_aware_qconfig


__all__ = [
    "QuantizableVGG",
    "VGG11_Weights",
    "VGG11_BN_Weights",
    "VGG13_Weights",
    "VGG13_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights",
    "VGG19_Weights",
    "VGG19_BN_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


class QuantizableVGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def fuse_model(self, is_qat: bool = False):
        fuse_vgg(self, is_qat=is_qat)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)

        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(
    cfg: str,
    batch_norm: bool,
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    is_qat: bool,
    **kwargs: Any
) -> QuantizableVGG:
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(
                kwargs, "num_classes", len(weights.meta["categories"])
            )
    model = QuantizableVGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    model.eval()

    if quantize:
        if is_qat:
            model.fuse_model(is_qat=is_qat)
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)

        else:
            model.fuse_model(is_qat=is_qat)
            model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            torch.ao.quantization.prepare(model, inplace=True)

    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
    "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
}


class VGG11_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11-8a719046.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132863336,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.020,
                    "acc@5": 88.628,
                }
            },
            "_ops": 7.609,
            "_file_size": 506.84,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG11_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132868840,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 70.370,
                    "acc@5": 89.810,
                }
            },
            "_ops": 7.609,
            "_file_size": 506.881,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg13-19584684.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 133047848,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.928,
                    "acc@5": 89.246,
                }
            },
            "_ops": 11.308,
            "_file_size": 507.545,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 133053736,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.586,
                    "acc@5": 90.374,
                }
            },
            "_ops": 11.308,
            "_file_size": 507.59,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16-397923af.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.592,
                    "acc@5": 90.382,
                }
            },
            "_ops": 15.47,
            "_file_size": 527.796,
        },
    )
    IMAGENET1K_FEATURES = Weights(
        # Weights ported from https://github.com/amdegroot/ssd.pytorch/
        url="https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            mean=(0.48235, 0.45882, 0.40784),
            std=(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),
        ),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "categories": None,
            "recipe": "https://github.com/amdegroot/ssd.pytorch#training-ssd",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": float("nan"),
                    "acc@5": float("nan"),
                }
            },
            "_ops": 15.47,
            "_file_size": 527.802,
            "_docs": """
                These weights can't be used for classification because they are missing values in the `classifier`
                module. Only the `features` module has valid values and can be used for feature extraction. The weights
                were trained using the original input standardization method as described in the paper.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138365992,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.360,
                    "acc@5": 91.516,
                }
            },
            "_ops": 15.47,
            "_file_size": 527.866,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 143667240,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.376,
                    "acc@5": 90.876,
                }
            },
            "_ops": 19.632,
            "_file_size": 548.051,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 143678248,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.218,
                    "acc@5": 91.842,
                }
            },
            "_ops": 19.632,
            "_file_size": 548.143,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", VGG11_Weights.IMAGENET1K_V1))
def vgg11(
    *,
    weights: Optional[VGG11_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG11_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG11_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG11_Weights
        :members:
    """
    weights = VGG11_Weights.verify(weights)

    return _vgg("A", False, weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG11_BN_Weights.IMAGENET1K_V1))
def vgg11_bn(
    *,
    weights: Optional[VGG11_BN_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-11-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG11_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG11_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG11_BN_Weights
        :members:
    """
    weights = VGG11_BN_Weights.verify(weights)

    return _vgg("A", True, weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG13_Weights.IMAGENET1K_V1))
def vgg13(
    *,
    weights: Optional[VGG13_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG13_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG13_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG13_Weights
        :members:
    """
    weights = VGG13_Weights.verify(weights)

    return _vgg("B", False, weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG13_BN_Weights.IMAGENET1K_V1))
def vgg13_bn(
    *,
    weights: Optional[VGG13_BN_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-13-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG13_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG13_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG13_BN_Weights
        :members:
    """
    weights = VGG13_BN_Weights.verify(weights)

    return _vgg("B", True, weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG16_Weights.IMAGENET1K_V1))
def vgg16(
    *,
    weights: Optional[VGG16_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG16_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG16_Weights
        :members:
    """
    weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG16_BN_Weights.IMAGENET1K_V1))
def vgg16_bn(
    *,
    weights: Optional[VGG16_BN_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG16_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG16_BN_Weights
        :members:
    """
    weights = VGG16_BN_Weights.verify(weights)

    return _vgg("D", True, weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG19_Weights.IMAGENET1K_V1))
def vgg19(
    *,
    weights: Optional[VGG19_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_Weights
        :members:
    """
    weights = VGG19_Weights.verify(weights)

    return _vgg("E", False, weights, progress, quantize, is_qat, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG19_BN_Weights.IMAGENET1K_V1))
def vgg19_bn(
    *,
    weights: Optional[VGG19_BN_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any
) -> QuantizableVGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize
        is_qat
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_BN_Weights
        :members:
    """
    weights = VGG19_BN_Weights.verify(weights)

    return _vgg("E", True, weights, progress, quantize, is_qat, **kwargs)


def fuse_vgg(model: nn.Module, version: str = "vgg16", is_qat: bool = False) -> None:
    if version == "vgg16":
        if isinstance(model.features[1], nn.BatchNorm2d):
            _fuse_modules(
                model.features,
                modules_to_fuse=[
                    ["0", "1", "2"],
                    ["3", "4", "5"],
                    ["7", "8", "9"],
                    ["10", "11", "12"],
                    ["14", "15", "16"],
                    ["17", "18", "19"],
                    ["20", "21", "22"],
                    ["24", "25", "26"],
                    ["27", "28", "29"],
                    ["30", "31", "32"],
                    ["34", "35", "36"],
                    ["37", "38", "39"],
                    ["40", "41", "42"],
                ],
                is_qat=is_qat,
                inplace=True,
            )
        else:
            _fuse_modules(
                model.features,
                modules_to_fuse=[
                    ["0", "1"],
                    ["2", "3"],
                    ["5", "6"],
                    ["7", "8"],
                    ["10", "11"],
                    ["12", "13"],
                    ["14", "15"],
                    ["17", "18"],
                    ["19", "20"],
                    ["21", "22"],
                    ["24", "25"],
                    ["26", "27"],
                    ["28", "29"],
                ],
                is_qat=is_qat,
                inplace=True,
            )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 224, 224)
    model_bn = vgg16_bn(quantize=True, is_qat=False)
    model = vgg16(quantize=True, is_qat=False)
    model_bn_fp = copy.deepcopy(model_bn)
    model_fp = copy.deepcopy(model)

    model(dummy_input)
    model_bn(dummy_input)
    dummy_output_fp = model_fp(dummy_input)
    dummy_output_bn_fp = model_bn_fp(dummy_input)

    torch.ao.quantization.convert(model_bn, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)

    dummy_output = model(dummy_input)
    dummy_output_bn = model_bn(dummy_input)

    mse = cal_mse(dummy_output_bn, dummy_output_bn_fp, norm=True)
    mse_bn = cal_mse(dummy_output, dummy_output_fp, norm=True)

    # from utils.onnx_utils import convert_onnx
    # convert_onnx(model_bn, "../onnx/vgg_bn_qint8.onnx", opset=13)  # success
    # convert_onnx(model_bn_fp, "../onnx/vgg_bn_fp32.onnx", opset=13)  # success
    # convert_onnx(model, "../onnx/vgg_qint8.onnx", opset=13)  # success
    # convert_onnx(model_fp, "../onnx/vgg_fp32.onnx", opset=13)  # success
