from typing import Union, Optional, Tuple
import torch
import torch.nn as nn


def convert_onnx(
    model: nn.Module,
    fname: str,
    opset: Union[int, None] = None,
    input_size: Optional[Union[None, Tuple[int, int, int, int]]] = None,
):
    if opset is None:
        opset = 13

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
