from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
from models.resnet import QuantizableResNet
from models.densenet import QuantizableDenseNet
from models.efficientnet import QuantizableEfficientNet
from models.yolo import QuantizableYoloBackbone
from models.mobilenetv2 import QuantizableMobileNetV2
from models.mobilenetv3 import QuantizableMobileNetV3


def convert_onnx(
    model: nn.Module,
    fname: str,
    opset: Union[int, None] = None,
    input_size: Optional[Union[None, Tuple[int, int, int, int]]] = None,
):
    if opset is None:
        opset = check_optimized_opset(model)

    if input_size is not None:
        if len(input_size) != 4:
            raise ValueError(
                f"len(input_size) == 4, but got length with {len(input_size)}"
            )
        dummy_input = torch.randn(*input_size)
    else:
        dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        f"{fname}",
        opset_version=opset,
    )


def check_optimized_opset(model: nn.Module) -> int:
    if model in [
        QuantizableResNet,
        QuantizableMobileNetV2,
        QuantizableMobileNetV3,
        QuantizableYoloBackbone,
    ]:
        return 13

    elif model in [QuantizableEfficientNet, QuantizableDenseNet]:
        raise AttributeError(f"{type(model)} can not be exported to onnx format. ")
