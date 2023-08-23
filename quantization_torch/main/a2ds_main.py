from typing import Tuple
import numpy as np
import torch
import onnx
import onnxruntime as ort

from quantization_torch.models.a2ds import resnet18, CRNN, RNNBase
from quantization_torch.utils.quantization_utils import cal_mse


def convert_cnn(input_shape: Tuple[int, int, int] = (1, 224, 224)):
    model = resnet18(num_classes=1000).eval()
    input_np = np.random.randn(1, *input_shape).astype(np.float32)
    dummy_input = torch.from_numpy(input_np)
    dummy_output = model(dummy_input)
    input_names = ["actual_input_1"] + [
        "learned_%d" % i for i in range(len(list(model.named_children())))
    ]
    output_names = ["output1"]

    torch.onnx.export(
        model,
        dummy_input,
        "../../onnx/a2ds_resnet18.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )

    onnx_model = onnx.load("../../onnx/a2ds_resnet18.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession("../../onnx/a2ds_resnet18.onnx")
    onnx_output = ort_session.run(
        None,
        {"actual_input_1": input_np},
    )
    mse = cal_mse(torch.from_numpy(onnx_output[0]), dummy_output, norm=False)
    print(f"[CNN] ONNX - TORCH PRECISION ERROR : {mse: .6f}")


def convert_rnn(input_shape: Tuple[int, int, int] = (1, 224, 224)):
    model = RNNBase(num_classes=1000).eval()
    input_np = np.random.randn(1, *input_shape).astype(np.float32)
    dummy_input = torch.from_numpy(input_np)
    dummy_output = model(dummy_input)
    input_names = ["actual_input_1"] + [
        "learned_%d" % i for i in range(len(list(model.named_children())))
    ]
    output_names = ["output1"]

    torch.onnx.export(
        model,
        dummy_input,
        "../../onnx/a2ds_rnn.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )

    onnx_model = onnx.load("../../onnx/a2ds_rnn.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession("../../onnx/a2ds_rnn.onnx")
    onnx_output = ort_session.run(
        None,
        {"actual_input_1": input_np},
    )
    mse = cal_mse(torch.from_numpy(onnx_output[0]), dummy_output, norm=False)
    print(f"[RNN] ONNX - TORCH PRECISION ERROR : {mse: .6f}")


def convert_crnn(input_shape: Tuple[int, int, int] = (1, 224, 224)):
    model = CRNN(num_classes=1000).eval()
    input_np = np.random.randn(1, *input_shape).astype(np.float32)
    dummy_input = torch.from_numpy(input_np)
    dummy_output = model(dummy_input)
    input_names = ["actual_input_1"] + [
        "learned_%d" % i for i in range(len(list(model.named_children())))
    ]
    output_names = ["output1"]

    torch.onnx.export(
        model,
        dummy_input,
        "../../onnx/a2ds_rcnn.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )

    onnx_model = onnx.load("../../onnx/a2ds_rcnn.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession("../../onnx/a2ds_rcnn.onnx")
    onnx_output = ort_session.run(
        None,
        {"actual_input_1": input_np},
    )
    mse = cal_mse(torch.from_numpy(onnx_output[0]), dummy_output, norm=False)
    print(f"[RCNN] ONNX - TORCH PRECISION ERROR : {mse: .6f}")


if __name__ == "__main__":
    convert_cnn()
    convert_rnn()
    convert_crnn()
