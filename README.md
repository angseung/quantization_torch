# Scope of Model Quantization & ONNX Exporting

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