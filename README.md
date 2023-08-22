# Model Quantization with PyTorch

# 1. 양자화 대상 모델

## 1.1. PyTorch

### 1.1.1. Classification

| Model | Quantized version is available in torchvision | Implemented Manually |
| --- | --- | --- |
| ResNet18 | Y | Y |
| ResNet34 | N | Y |
| ResNet50 | Y | Y |
| ResNet101 | N | Y |
| ResNet152 | N | Y |
| DenseNet121 | N | Y |# 1. 양자화 대상 모델

## 1.1. PyTorch

### 1.1.1. Classification

| Model | Quantized version is available in torchvision | Implemented Manually |
| --- | --- | --- |
| ResNet18 | Y | Y |
| ResNet34 | N | Y |
| ResNet50 | Y | Y |
| ResNet101 | N | Y |
| ResNet152 | N | Y |
| DenseNet121 | N | Y |
| DenseNet161 | N | Y |
| DenseNet169 | N | Y |
| DenseNet201 | N | Y |
| EfficientNetB0 | N | forward error |
| EfficientNetB1 | N | forward error |
| EfficientNetB2 | N | forward error |
| EfficientNetB3 | N | forward error |
| EfficientNetB4 | N | forward error |
| EfficientNetB5 | N | forward error |
| EfficientNetB6 | N | forward error |
| EfficientNetB7 | N | forward error |
| MobileNetV2 | Y | Y |
| MobileNetV3 | Y | Y |
| WideResNet50 | N | Y |
| WideResNet101 | N | Y |
| ResNext50 | N | Y |
| ResNext101 | N | Y |
| SqueezeNet 1.0 | N | Y |
| SqueezeNet 1.1 | N | Y |
| MNASNet 0.5 | N | Y |
| MNASNet 0.75 | N | Y |
| MNASNet 1.0 | N | Y |
| MNASNet 1.3 | N | Y |
| ConvNeXt_tiny | N | forward error |
| ConvNeXt_small | N | forward error |
| ConvNeXt_base | N | forward error |
| ConvNeXt_large | N | forward error |
| RegNet_X_400mf | N | Y |
| RegNet_X_800mf | N | Y |
| RegNet_X_1.6gf | N | Y |
| RegNet_X_3.2gf | N | Y |
| RegNet_X_8gf | N | Y |
| RegNet_X_16gf | N | Y |
| RegNet_X_32gf | N | Y |
| RegNet_Y_400mf | N | Y |
| RegNet_Y_800mf | N | Y |
| RegNet_Y_1.6gf | N | Y |
| RegNet_Y_3.2gf | N | Y |
| RegNet_Y_8gf | N | Y |
| RegNet_Y_16gf | N | Y |
| RegNet_Y_32gf | N | Y |
| RegNet_Y_128gf | N | Y |
| VGG11 | N | Y |
| VGG11_BN | N | Y |
| VGG13 | N | Y |
| VGG13_BN | N | Y |
| VGG16 | N | Y |
| VGG16_BN | N | Y |
| VGG19 | N | Y |
| VGG19_BN | N | Y |

 

### 1.1.2. Detection

| Model | Quantized version is available in torchvision | Implemented Manually |
| --- | --- | --- |
| YoloV3 | N | Y |
| YoloV4 | N | Y |
| YoloV5n | N | Y |
| YoloV5m | N | Y |
| YoloV5l | N | Y |
| YoloV5x | N | Y |
| RetinaNet_ResNet50_FPN | N | Y |
| RetinaNet_ResNet50_FPN_V2 | N | Y |
| SSD300_VGG16 | N | Y |
| SSDLite320_MobileNetV3_Large | N | Y |
| FCOS_ResNet50_FPN | N | Y |
| Faster R-CNN_ResNet50_FPN | N | Y |
| Faster R-CNN_ResNet50_FPN_V2 | N | Y |
| Faster R-CNN_MobileNetV3_Large_320_FPN | N | Y |
| Faster R-CNN_MobileNetV3_Large_320 | N | Y |

### 1.1.3. Segmentation

| Model | Quantized version is available in torchvision | Implemented Manually |
| --- | --- | --- |
| FCN_ResNet50 | N | Y |
| FCN_ResNet101 | N | Y |
| DeepLabV3_MobileNetV3 | N | Y |
| DeepLabV3_ResNet50 | N | Y |
| DeepLabV3_ResNet101 | N | Y |
| LRASPP_MobileNetV3_Large | N | Y |
| Mask R-CNN_ResNet50_FPN | N | N |
| Mask R-CNN_ResNet50_FPN_V2 | N | N |

## 1.1.4. Issue

- EfficientNet 계열: Stride ≠ 1인 Pointwise Convolution의 qint8 연산 미지원 (forward error)
- ConvNext 계열: GeLU 활성화 함수 및 Layer Normalization의 qint8 연산 미지원 (forward error)

- Pointwise Convolution에서 qint8 자료형 지원 불가 문제 (forward error)
    - stride ≠ 1인 Pointwise Convolution을 사용하는 EfficientNet, EfficientDet, MobileNetV1 등 모델들은 PyTorch에서 양자화 적용 후 추론 불가
    - Depthwise Convolution은 stride에 관계 없이 양자화 및 Fusing 가능

https://github.com/pytorch/pytorch/issues/74540

- Tensor Element-wise Multiply 연산 미지원 문제
    - Tensor-Tensor 곱 연산은 FloatFunctional.mul 메서드로 변경하여 해결 가능

- Faster R-CNN 계열: ARM 아키텍쳐에서 양자화 버전 실행 불가
    - QNNPACK의 Maxpooling 연산 Assertion Error

```bash
RuntimeError: createStatus == pytorch_qnnp_status_success INTERNAL ASSERT FAILED at "/home/***/***/pytorch/aten/src/ATen/native/quantized/cpu/Pooling.cpp":328, please report a bug to PyTorch. failed to create QNNPACK MaxPool operator
```

- Mask R-CNN의 Transposed Convolution은 qnnpack에서만 양자화 가능하며, 이는 ARM 아키텍쳐에서만 실행 가능함

# 2. 추론시간 벤치마크

## 2.1. 1차 양자화 대상 모델 추론시간 비교

- DenseNet 계열
    - Fusing이 불가능하며, AArch64에서는 양자화 이후에도 추론시간이 오히려 증가할 수 있음
    - DenseNet은 양자화를 적용하면 ONNX Export가 불가능함

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| DenseNet121 | 1.78 | 1.39 | 0.78 |
| DenseNet161 | 2.87 | 2.74 | 0.95 |
| DenseNet169 | 2.37 | 1.71 | 0.72 |
| DenseNet201 | 2.84 | 2.19 | 0.77 |

- ResNet 계열

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| ResNet18 | 0.58 | 0.35 | 0.61 |
| ResNet34 | 1.07 | 0.52 | 0.49 |
| ResNet50 | 1.27 | 0.6 | 0.47 |
| ResNet101 | 2.19 | 1.05 | 0.48 |
| ResNet152 | 3.02 | 1.38 | 0.46 |
| ResNext50 | 1.6 | 0.58 | 0.36 |
| ResNext101 | 4.58 | 1.56 | 0.34 |
| WideResNet50 | 2.56 | 1.02 | 0.4 |
| WideResNet101 | 4.81 | 1.91 | 0.4 |

- MobileNet 계열

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| MobileNetV2 | 1.42 | 0.75 | 0.53 |
| MobileNetV3 | 1.67 | 0.75 | 0.44 |

- Yolo 계열

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| YoloV3 | 2.111 | 1.010 | 0.48  |
| YoloV4 | 4.367 | 1.433 | 0.33  |
| YoloV5n |  |  |  |
| YoloV5m | 0.777 | 0.332 | 0.43  |
| Yolov5l | 1.617 | 0.677 | 0.42  |
| YoloV5x |  |  |  |

## 2.2. 2차 양자화 대상 모델 추론시간 비교

| Target | Intel core i5-12500 |
| --- | --- |
| Arch | x86-64 |
| OS | Windows 11 |

- Classification 모델

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| SqueezeNet 1.0 |  |  |  |
| SqueezeNet 1.1 |  |  |  |
| MNASNet 0,5 |  |  |  |
| MNASNet 0,75 |  |  |  |
| MNASNet 1.0 |  |  |  |
| MNASNet 1.3 |  |  |  |
| ConvNeXt_tiny |  | N/A | N/A |
| ConvNeXt_small |  | N/A | N/A |
| ConvNeXt_base |  | N/A | N/A |
| ConvNeXt_large |  | N/A | N/A |
| RegNet_X_400mf |  |  |  |
| RegNet_X_800mf |  |  |  |
| RegNet_X_1.6gf |  |  |  |
| RegNet_X_3.2gf |  |  |  |
| RegNet_X_8gf |  |  |  |
| RegNet_X_16gf |  |  |  |
| RegNet_X_32gf |  |  |  |
| RegNet_Y_400mf |  |  |  |
| RegNet_Y_800mf |  |  |  |
| RegNet_Y_1.6gf |  |  |  |
| RegNet_Y_3.2gf |  |  |  |
| RegNet_Y_8gf |  |  |  |
| RegNet_Y_16gf |  |  |  |
| RegNet_Y_32gf |  |  |  |
| RegNet_Y_128gf |  |  |  |
| VGG11 |  |  |  |
| VGG11_BN |  |  |  |
| VGG13 |  |  |  |
| VGG13_BN |  |  |  |
| VGG16 |  |  |  |
| VGG16_BN |  |  |  |
| VGG19 |  |  |  |
| VGG19_BN |  |  |  |

- Detection 모델

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| RetinaNet_ResNet50_FPN | 193.8 | 23.73 |  |
| RetinaNet_ResNet50_FPN_V2 | 193.4 | 24.61 |  |
| SSD300_VGG16 |  |  |  |
| SSDLite320_MobileNetV3_Large |  |  |  |
| FCOS_ResNet50_FPN |  |  |  |
| Faster R-CNN_ResNet50_FPN |  | QNNPACK ERROR | - |
| Faster R-CNN_ResNet50_FPN_V2 |  | QNNPACK ERROR | - |
| Faster R-CNN_MobileNetV3_Large_320_FPN |  | QNNPACK ERROR | - |
| Faster R-CNN_MobileNetV3_Large_320 |  | QNNPACK ERROR | - |

- Segmentation 모델

| Model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| FCN_ResNet50 |  |  |  |
| FCN_ResNet101 |  |  |  |
| DeepLabV3_MobileNetV3 |  |  |  |
| DeepLabV3_ResNet50 |  |  |  |
| DeepLabV3_ResNet101 |  |  |  |
| LRASPP_MobileNetV3_Large |  |  |  |
| Mask R-CNN_ResNet50_FPN |  | QNNPACK ERROR | - |
| Mask R-CNN_ResNet50_FPN_V2 |  | QNNPACK ERROR | - |

# 3. ONNX Export Test

## 3.1. 1차 지원 모델군

| Model | Pretrained | Quantization Support | FP32 model ONNX Export Support | Quantized model ONNX Export Support | ONNX opset |
| --- | --- | --- | --- | --- | --- |
| ResNet18 | Y | PTQ, QAT | Y | Y | 13 |
| ResNet34 | Y | PTQ, QAT | Y | Y | 13 |
| ResNet50 | Y | PTQ, QAT | Y | Y | 13 |
| ResNet101 | Y | PTQ, QAT | Y | Y | 13 |
| ResNet152 | Y | PTQ, QAT | Y | Y | 13 |
| DenseNet121 | Y | PTQ, QAT | Y | N | N/A |
| DenseNet161 | Y | PTQ, QAT | Y | N | N/A |
| DenseNet169 | Y | PTQ, QAT | Y | N | N/A |
| DenseNet201 | Y | PTQ, QAT | Y | N | N/A |
| EfficientNetB0 | Y | N | Y | N | 13 |
| EfficientNetB1 | Y | N | Y | N | 13 |
| EfficientNetB2 | Y | N | Y | N | 13 |
| EfficientNetB3 | Y | N | Y | N | 13 |
| EfficientNetB4 | Y | N | Y | N | 13 |
| EfficientNetB5 | Y | N | Y | N | 13 |
| EfficientNetB6 | Y | N | Y | N | 13 |
| EfficientNetB7 | Y | N | Y | N | 13 |
| MobileNetV2 | Y | PTQ, QAT | Y | Y | 13 |
| MobileNetV3 | Y | PTQ, QAT | Y | Y | 13 |
| WideResNet50 | Y | PTQ, QAT | Y | Y | 13 |
| WideResNet101 | Y | PTQ, QAT | Y | Y | 13 |
| ResNext50 | Y | PTQ, QAT | Y | Y | 13 |
| ResNext101 | Y | PTQ, QAT | Y | Y | 13 |
| YoloV3 | N | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
| YoloV4 | N | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
| YoloV5n | N | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
| YoloV5m | N | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
| YoloV5l | N | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
| YoloV5x | N | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |
- YoloV3~V5: Swish → ReLU 활성으로 변경함에 따라 기존 모델 가중치 호환성 이슈 존재

## 3.2. 2차 지원 모델군

| Model | Pretrained | Quantization Support | FP32 model ONNX Export Support | Quantized model ONNX Export Support | ONNX opset |
| --- | --- | --- | --- | --- | --- |
| SqueezeNet 1.0 | Y | PTQ, QAT | Y | Y | 13 |
| SqueezeNet 1.1 | Y | PTQ, QAT | Y | Y | 13 |
| MNASNet_0,5 | Y | PTQ, QAT | Y | Y | 13 |
| MNASNet_0,75 | Y | PTQ, QAT | Y | Y | 13 |
| MNASNet_1.0 | Y | PTQ, QAT | Y | Y | 13 |
| MNASNet_1.3 | Y | PTQ, QAT | Y | Y | 13 |
| ConvNeXt_tiny | Y | N | Y | N | 13 |
| ConvNeXt_small | Y | N | Y | N | 13 |
| ConvNeXt_base | Y | N | Y | N | 13 |
| ConvNeXt_large | Y | N | Y | N | 13 |
| RegNet_X_400mf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_X_800mf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_X_1.6gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_X_3.2gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_X_8gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_X_16gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_X_32gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_400mf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_800mf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_1.6gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_3.2gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_8gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_16gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_32gf | Y | PTQ, QAT | Y | N | 13 |
| RegNet_Y_128gf | Y | PTQ, QAT | Y | N | 13 |
| RetinaNet_ResNet50_FPN | Y | PTQ, QAT | Y | N | 13 |
| RetinaNet_ResNet50_FPN_V2 | Y | PTQ, QAT | Y | N | 13 |
| VGG16 | Y | PTQ, QAT | Y | Y | 13 |
| VGG16_BN | Y | PTQ, QAT | Y | Y | 13 |
| SSD300_VGG16 | Y | PTQ, QAT | Y | N | 13 |
| SSDLite320_MobileNetV3_Large | N | PTQ, QAT | Y | N | 13 |
| FCOS_ResNet50_FPN | Y | PTQ, QAT | Y | N | 13 |
| Faster R-CNN_ResNet50_FPN | Y | PTQ, QAT | Y | N | 13 |
| Faster R-CNN_ResNet50_FPN_V2 | Y | PTQ, QAT | Y | N | 13 |
| Faster R-CNN_MobileNetV3_Large_320_FPN | Y | PTQ, QAT | Y | N | 13 |
| Faster R-CNN_MobileNetV3_Large_320 | Y | PTQ, QAT | Y | N | 13 |
| FCN_ResNet50 | Y | PTQ, QAT | Y | N | 13 |
| FCN_ResNet101 | Y | PTQ, QAT | Y | N | 13 |
| DeepLabV3_MobileNetV3 | Y | PTQ, QAT | Y | N | 13 |
| DeepLabV3_ResNet50 | Y | PTQ, QAT | Y | N | 13 |
| DeepLabV3_ResNet101 | Y | PTQ, QAT | Y | N | 13 |
| Mask R-CNN_ResNet50_FPN | Y | PTQ, QAT (ARM Only) | Y | N | 13 |
| Mask R-CNN_ResNet50_FPN_V2 | Y | PTQ, QAT (ARM Only) | Y | N | 13 |
| LRASPP_MobileNetV3_Large | Y | PTQ, QAT | Y | N | 13 |
- SSDLite: ReLU6 → ReLU로 변경함에 따라 기존 모델 가중치 호환성 이슈 존재
- Mask R-CNN 계열: ARM 아키텍쳐에서만 양자화 기능 지원
    - 단, Faster R-CNN 계열과 마찬가지로 QNNPACK ERROR로 인해 추론 불가