import torch
from torchvision import models
import onnx
import onnxruntime

from utils.onnx_utils import convert_onnx
from utils.quantization_utils import cal_mse

models_to_test = [
    models.resnet18(),
    models.resnet34(),
    models.resnet50(),
    models.resnet101(),
    models.resnet152(models.ResNet152_Weights),
    models.resnext50_32x4d(),
    models.resnext101_32x8d(),
    models.resnext101_64x4d(),
    models.wide_resnet50_2(),
    models.wide_resnet101_2(),
    models.efficientnet_b0(),
    models.efficientnet_b1(),
    models.efficientnet_b2(),
    models.efficientnet_b3(),
    models.efficientnet_b4(),
    models.efficientnet_b5(),
    models.efficientnet_b6(),
    models.efficientnet_b7(),
    models.densenet121(),
    models.densenet161(),
    models.densenet169(),
    models.densenet201(),
    models.mobilenet_v2(),
    models.mobilenet_v3_large(),
    models.mobilenet_v3_small(),
]

if __name__ == "__main__":
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)

    for model in models_to_test:
        model.eval()
        torch_output = model(dummy_input).detach().cpu().numpy()

        # convert model to onnx and inference with it
        convert_onnx(
            model=model,
            fname=f"./onnx/{model.__class__.__name__}_test.onnx",
            input_size=input_shape,
            opset=13,
        )
        onnx.checker.check_model(f"./onnx/{model.__class__.__name__}_test.onnx", True)

        onnx_model = onnxruntime.InferenceSession(
            f"./onnx/{model.__class__.__name__}_test.onnx"
        )
        onnx_inputs = {
            onnx_model.get_inputs()[0].name: dummy_input.detach().cpu().numpy()
        }
        onnx_output = onnx_model.run(None, onnx_inputs)[0]
        nmse = cal_mse(
            torch.from_numpy(onnx_output), torch.from_numpy(torch_output), norm=False
        )

        if nmse >= 1e-3:
            print(f"{model.__class__.__name__} failed, mse: {nmse: .6f}")
        else:
            print(f"{model.__class__.__name__} done, mse: {nmse: .6f}")
