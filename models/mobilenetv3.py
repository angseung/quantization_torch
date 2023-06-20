"""
it overrides torchvision.models.quantization.mobilenetv3
"""

from functools import partial
from typing import Any, List, Optional, Union

import torch
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.transforms._presets import ImageClassification
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.quantization.mobilenetv3 import (
    _mobilenet_v3_conf,
    InvertedResidual,
    InvertedResidualConfig,
    MobileNet_V3_Large_Weights,
    MobileNetV3,
)
from torchvision.models.quantization.utils import _fuse_modules, _replace_relu
from utils.quantization_utils import get_platform_aware_qconfig, cal_mse
from utils.onnx_utils import convert_onnx

__all__ = [
    "QuantizableMobileNetV3",
    "MobileNet_V3_Large_QuantizedWeights",
    "mobilenet_v3_large",
]


class QuantizableSqueezeExcitation(SqueezeExcitation):
    _version = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["scale_activation"] = nn.Hardsigmoid
        super().__init__(*args, **kwargs)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, input: Tensor) -> Tensor:
        return self.skip_mul.mul(self._scale(input), input)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(self, ["fc1", "activation"], is_qat, inplace=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if hasattr(self, "qconfig") and (version is None or version < 2):
            default_state_dict = {
                "scale_activation.activation_post_process.scale": torch.tensor([1.0]),
                "scale_activation.activation_post_process.activation_post_process.scale": torch.tensor(
                    [1.0]
                ),
                "scale_activation.activation_post_process.zero_point": torch.tensor(
                    [0], dtype=torch.int32
                ),
                "scale_activation.activation_post_process.activation_post_process.zero_point": torch.tensor(
                    [0], dtype=torch.int32
                ),
                "scale_activation.activation_post_process.fake_quant_enabled": torch.tensor(
                    [1]
                ),
                "scale_activation.activation_post_process.observer_enabled": torch.tensor(
                    [1]
                ),
            }
            for k, v in default_state_dict.items():
                full_key = prefix + k
                if full_key not in state_dict:
                    state_dict[full_key] = v

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class QuantizableInvertedResidual(InvertedResidual):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, se_layer=QuantizableSqueezeExcitation, **kwargs)  # type: ignore[misc]
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.block(x))
        else:
            return self.block(x)


class QuantizableMobileNetV3(MobileNetV3):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V3 main class

        Args:
           Inherits args from floating point MobileNetV3
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for m in self.modules():
            if type(m) is Conv2dNormActivation:
                modules_to_fuse = ["0", "1"]
                if len(m) == 3 and type(m[2]) is nn.ReLU:
                    modules_to_fuse.append("2")
                _fuse_modules(m, modules_to_fuse, is_qat, inplace=True)
            elif type(m) is QuantizableSqueezeExcitation:
                m.fuse_model(is_qat)


def _mobilenet_v3_model(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    is_qat: bool,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])

    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    model = QuantizableMobileNetV3(
        inverted_residual_setting,
        last_channel,
        block=QuantizableInvertedResidual,
        **kwargs,
    )
    _replace_relu(model)
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


class MobileNet_V3_Large_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v3_large_qnnpack-5bcacf28.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 5483032,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "backend": "qnnpack",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv3",
            "unquantized": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.004,
                    "acc@5": 90.858,
                }
            },
            "_ops": 0.217,
            "_file_size": 21.554,
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
        lambda kwargs: MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    )
)
def mobilenet_v3_large(
    *,
    weights: Optional[
        Union[MobileNet_V3_Large_QuantizedWeights, MobileNet_V3_Large_Weights]
    ] = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    """
    MobileNetV3 (Large) model from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights` or :class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool): If True, return a quantized version of the model. Default is False.
        is_qat:
        **kwargs: parameters passed to the ``torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights
        :members:
    .. autoclass:: torchvision.models.MobileNet_V3_Large_Weights
        :members:
        :noindex:
    """
    weights = (
        MobileNet_V3_Large_QuantizedWeights if quantize else MobileNet_V3_Large_Weights
    ).verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large", **kwargs
    )
    return _mobilenet_v3_model(
        inverted_residual_setting,
        last_channel,
        weights,
        progress,
        quantize,
        is_qat,
        **kwargs,
    )


if __name__ == "__main__":
    import copy

    model = mobilenet_v3_large(quantize=True, is_qat=True)
    model_fp = copy.deepcopy(model)
    input = torch.randn(1, 3, 224, 224)
    model(input)  # Calibration codes here...
    torch.ao.quantization.convert(model, inplace=True)
    dummy_output = model(input)
    dummy_output_fp = model_fp(input)
    nmse = cal_mse(dummy_output, dummy_output_fp, norm=False)
