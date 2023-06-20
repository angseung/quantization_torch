# Models which Support Quantization & ONNX Exporting

| Model | Quantization Support | FP32 model ONNX Export Support | Quantized model ONNX Export Support | ONNX opset |
| --- | --- | --- | --- | --- |
| ResNet18 | PTQ, QAT | Y | Y | 13 |
| ResNet34 | PTQ, QAT | Y | Y | 13 |
| ResNet50 | PTQ, QAT | Y | Y | 13 |
| ResNet101 | PTQ, QAT | Y | Y | 13 |
| ResNet152 | PTQ, QAT | Y | Y | 13 |
| DenseNet121 | PTQ, QAT | Y | N | N/A |
| DenseNet161 | PTQ, QAT | Y | N | N/A |
| DenseNet169 | PTQ, QAT | Y | N | N/A |
| DenseNet201 | PTQ, QAT | Y | N | N/A |
| EfficientNetB0 | N | Y | N | N/A |
| EfficientNetB1 | N | Y | N | N/A |
| EfficientNetB2 | N | Y | N | N/A |
| EfficientNetB3 | N | Y | N | N/A |
| EfficientNetB4 | N | Y | N | N/A |
| EfficientNetB5 | N | Y | N | N/A |
| EfficientNetB6 | N | Y | N | N/A |
| EfficientNetB7 | N | Y | N | N/A |
| MobileNetV2 | PTQ, QAT | Y | Y | 13 |
| MobileNetV3 | PTQ, QAT | Y | Y | 13 |
| WideResNet50 | PTQ, QAT | Y | Y | 13 |
| WideResNet101 | PTQ, QAT | Y | Y | 13 |
| ResNext50 | PTQ, QAT | Y | Y | 13 |
| ResNext101 | PTQ, QAT | Y | Y | 13 |
| YoloV3 | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
| YoloV4 | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
| YoloV5 | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |

# 양자화 대상 모델 추론시간 비교 (on Raspberry Pi 4B 2GB Model)

- DenseNet 계열
    - Fusing이 불가능하며, AArch64에서는 양자화 이후에도 추론시간이 오히려 증가할 수 있음
    - DenseNet은 양자화를 적용하면 ONNX Export가 불가능함

| model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| densenet121 | 1.78 | 1.39 | 0.78 |
| densenet161 | 2.87 | 2.74 | 0.95 |
| densenet169 | 2.37 | 1.71 | 0.72 |
| densenet201 | 2.84 | 2.19 | 0.77 |
- ResNet 계열

| model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| resnet18 | 0.58 | 0.35 | 0.61 |
| resnet34 | 1.07 | 0.52 | 0.49 |
| resnet50 | 1.27 | 0.6 | 0.47 |
| resnet101 | 2.19 | 1.05 | 0.48 |
| resnet152 | 3.02 | 1.38 | 0.46 |
| resnext50_32x4d | 1.6 | 0.58 | 0.36 |
| resnext101_32x8d | 4.58 | 1.56 | 0.34 |
| wide_resnet50_2 | 2.56 | 1.02 | 0.4 |
| wide_resnet101_2 | 4.81 | 1.91 | 0.4 |
- MobileNet 계열

| model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| mobilenet_v2 | 1.42 | 0.75 | 0.53 |
| mobilenet_v3 | 1.67 | 0.75 | 0.44 |

- Yolo 계열

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| YoloV3 | 2.111 | 1.010 | 0.48  |
| YoloV4 | 4.367 | 1.433 | 0.33  |
| YoloV5m | 0.777 | 0.332 | 0.43  |
| Yolov5l | 1.617 | 0.677 | 0.42  |

# Design Principle for Standard Wrapper Class of Quantizable Model

# 1. Quantizable Model Class

```python
class QuantizableSomeModel(SomeModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
```

- 양자화 가능한 네트워크인 SomeModel의 양자화를 위한 모델 클래스명은 QuantizableSomeModel 으로 하고, SomeModel을 상속한다.
    - QuantizableSomeModel 인스턴스라 할지라도, torch.ao.quantization.convert 함수를 통해 양자화를 수행하지 않으면 실수형 모델이다.
    - QuantStub, DeQuantStub이 forward를 감싸고 있어도, convert를 하지 않으면 양자화 하지 않은 SomeModel과 완전히 동등한 결과를 얻는다.
- forward 메서드에서 quant, dequant를 포함한다.
- fuse_model 메서드는 모델의 구조에 따라 각각 구현하여야 하므로, SomeModel을 상속한 QuantizableSomeModel 클래스의 메서드로 한다.

# 2. Model Parser Function

```python
def somemodel(
    *,
    weights: = None,
    progress: bool = True,
    quantize: bool = False,
    is_qat: bool = False,
    **kwargs: Any,
) -> SomeModel:

    return _somemodel(
        inverted_residual_setting,
        last_channel,
        weights,
        progress,
        quantize,
        is_qat,
        **kwargs,
    )
```

- 동일한 모델이라도 Config에 따라 다양한 크기의 모델을 얻을 수 있다.
    - ex) ResNet 계열에서는 ResNet18, ResNet34, ResNet52 등 다양한 크기의 모델이 있다.
- Model Parser는 funtion이며, somemodel로 명명한다.
- Model Config을 받아 Model을 Build하는 function은 _somemodel()로 분리한다.

# 3. Model Builder Function

```python
def _somemodel(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    is_qat: bool,
    **kwargs: Any,
) -> SomeModel:
		
    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    model = SomeModel(
        inverted_residual_setting,
        last_channel,
        block=QuantizableInvertedResidual,
        **kwargs,
    )
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
```

- Model Builder는 Config을 읽어 모델 인스턴스를 반환한다.
- Quantizable 모델을 위해 quantize, is_qat를 Args로 받는다.
    - quantize: PTQ 또는 QAT를 할 때 True로 set
        - QAT를 할 때 is_qat를 True로 set
        - PTQ를 할 때 is_qat를 False로 set
    - fuse_module 함수 및 prepare 메서드가 QAT일 때와 PTQ일 때 동작이 다르기 때문에 PTQ와 QAT를 구별해야 한다.
- Fusing → Weights Load → Prepare 순서대로 수행한다.
    - 이는 Weights의 Quantization Error를 줄여 Precision을 유지하기 위함이다.
    - 단, Weights Load는 fuse_model로 레이어 Fusing 한 모델의 Weight만 로딩 가능하다.
    - Fusing 이전의 모델의 Weights를 로드하는 경우에는 모델을 fuse_model 메서드로 fusing한 뒤에 load_state_dict를 수행한다.

# 4. 모델 양자화 시퀀스

- PTQ

```python
# build full-precision model and prepare for PTQ
model = somemodel(config, quantize=True, is_qat=False)

# prepare dataloader for calibration
input = torch.randn(1, 3, 224, 224)
model(input)  # Calibration codes here...

# Quantize model after calibration
torch.ao.quantization.convert(model, inplace=True)

# get prediction
dummy_output = model(input)
```

- QAT

```python
# build full-precision model and prepare for QAT
model = somemodel(config, quantize=True, is_qat=True)

# train the model, this is Quantization Aware Training with fake quantization observers
train(model, dataset)

# Quantize model after training
torch.ao.quantization.convert(model, inplace=True)

# get prediction
dummy_output = model(input)
```
