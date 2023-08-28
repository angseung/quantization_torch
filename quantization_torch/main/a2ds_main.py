from typing import Tuple
import numpy as np
import torch
import onnx
import onnxruntime as ort

from quantization_torch.models.a2ds import resnet18, CRNN, RNNBase
from quantization_torch.utils.quantization_utils import cal_mse


def convert_cnn(
    input_shape: Tuple[int, int, int] = (1, 224, 224), num_classes: int = 1000
):
    if input_shape[0] == 3:
        # overrides resnet18
        from torchvision.models.resnet import resnet18 as resnet18_ori

        model = resnet18_ori(num_classes=num_classes).eval()
    else:
        model = resnet18(num_classes=num_classes).eval()

    input_np = np.random.randn(1, *input_shape).astype(np.float32)
    dummy_input = torch.from_numpy(input_np)
    dummy_output = model(dummy_input)
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        "../../onnx/a2ds_resnet18.onnx",
        opset_version=13,
        do_constant_folding=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    onnx_model = onnx.load("../../onnx/a2ds_resnet18.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession("../../onnx/a2ds_resnet18.onnx")
    onnx_output = ort_session.run(
        None,
        {"input": input_np},
    )
    mse = cal_mse(torch.from_numpy(onnx_output[0]), dummy_output, norm=False)
    print(f"[CNN] ONNX - TORCH PRECISION ERROR : {mse: .6f}")


def convert_rnn(input_shape: int = 256, num_classes: int = 87, rnn_type: str = "lstm"):
    model = RNNBase(
        input_size=input_shape, batch_size=1, rnn_type=rnn_type, num_classes=num_classes
    ).eval()
    input_np = np.random.randn(1, 1, input_shape).astype(np.float32)
    dummy_input = torch.from_numpy(input_np)
    dummy_output = model(dummy_input)
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        f"../../onnx/a2ds_rnn_{rnn_type}.onnx",
        opset_version=13,
        do_constant_folding=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    onnx_model = onnx.load(f"../../onnx/a2ds_rnn_{rnn_type}.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(f"../../onnx/a2ds_rnn_{rnn_type}.onnx")
    onnx_output = ort_session.run(
        None,
        {"input": input_np},
    )
    mse = cal_mse(torch.from_numpy(onnx_output[0]), dummy_output, norm=False)
    print(f"[RNN] ONNX - TORCH PRECISION ERROR : {mse: .6f}")


def convert_crnn(input_shape: Tuple[int, int] = (128, 256), num_classes: int = 87):
    model = CRNN(num_classes=num_classes, input_size=input_shape, batch_size=1).eval()
    input_np = np.random.randn(1, 1, *input_shape).astype(np.float32)
    dummy_input = torch.from_numpy(input_np)
    dummy_output = model(dummy_input)
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        "../../onnx/a2ds_crnn.onnx",
        opset_version=13,
        do_constant_folding=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    onnx_model = onnx.load("../../onnx/a2ds_crnn.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession("../../onnx/a2ds_crnn.onnx")
    onnx_output = ort_session.run(
        None,
        {"input": input_np},
    )
    mse = cal_mse(torch.from_numpy(onnx_output[0]), dummy_output, norm=False)
    print(f"[CRNN] ONNX - TORCH PRECISION ERROR : {mse: .6f}")


if __name__ == "__main__":
    convert_cnn(input_shape=(1, 224, 224), num_classes=50)
    convert_crnn(input_shape=(128, 256), num_classes=50)
    convert_rnn(input_shape=256, num_classes=87, rnn_type="lstm")
    convert_rnn(input_shape=256, num_classes=87, rnn_type="gru")
