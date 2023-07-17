# 1. PTQ/QAT 대상으로 선정한 모델

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
| EfficientNetB0 | N | 양자화 미지원 |
| EfficientNetB1 | N | 양자화 미지원 |
| EfficientNetB2 | N | 양자화 미지원 |
| EfficientNetB3 | N | 양자화 미지원 |
| EfficientNetB4 | N | 양자화 미지원 |
| EfficientNetB5 | N | 양자화 미지원 |
| EfficientNetB6 | N | 양자화 미지원 |
| EfficientNetB7 | N | 양자화 미지원 |
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
| ConvNeXt | N | 양자화 미지원 |
| RegNet X | N | Y |
| RegNet Y | N | 양자화 미지원 |
| VGG | N | Y |
| VGG_BN | N | Y |
- EfficientNet 계열: Stride ≠ 1인 Pointwise Convolution의 qint8 연산 미지원
- ConvNext 계열: GeLU 활성화 함수 및 Layer Normalization의 qint8 연산 미지원
- RegNet Y 계열: SEBlock Tensor-Mul의 qint8 연산 미지원

 

### 1.1.2. Detection

| Model | Quantized version is available in torchvision | Implemented Manually |
| --- | --- | --- |
| YoloV3 | N | Y |
| YoloV4 | N | Y |
| YoloV5 | N | Y |
| RetinaNet_ResNet50_FPN | N | Y |
| RetinaNet_ResNet50_FPN_V2 | N | Y |
| SSD300_VGG16 | N | Y |
| SSDLite320_MobileNetV3_Large | N | Y |
| FCOS | N | Y |

### 1.1.3. Segmentation

| Model | Quantized version is available in torchvision | Implemented Manually |
| --- | --- | --- |
| FPN | N |  |
| DeepLabV3 | N |  |
| LRASPP | N |  |
| Mask R-CNN | N |  |

## 1.1.4. Issue

- Pointwise Convolution에서 qint8 자료형 지원 불가 문제
    - stride ≠ 1인 Pointwise Convolution을 사용하는 EfficientNet, EfficientDet, MobileNetV1 등 모델들은 PyTorch에서 양자화 적용 불가
    - Depthwise Convolution은 stride에 관계 없이 양자화 및 Fusing 가능

https://github.com/pytorch/pytorch/issues/74540

- Tensor Element-wise Multiply 연산 미지원 문제

# 2. 추론시간 벤치마크

## 2.1. 1차 양자화 대상 모델 추론시간 비교

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

## 2.2. 2차 양자화 대상 모델 추론시간 비교

- Classification 모델

| model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| SqueezeNet 1.0 |  |  |  |
| SqueezeNet 1.1 |  |  |  |
| MNASNet_0,5 |  |  |  |
| MNASNet_0,75 |  |  |  |
| MNASNet_1.0 |  |  |  |
| MNASNet_1.3 |  |  |  |
| ConvNeXt |  |  |  |
| RegNet X |  |  |  |
| RegNet Y |  |  |  |
| VGG16 |  |  |  |
| VGG16_BN |  |  |  |

- Detection 모델

| model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| RetinaNet_ResNet50_FPN | 193.8 | 23.73 |  |
| RetinaNet_ResNet50_FPN_V2 | 193.4 | 24.61 |  |
| SSD300_VGG16 |  |  |  |
| SSDLite320_MobileNetV3_Large |  |  |  |
| FCOS |  |  |  |

- Segmentation 모델

| model | before (s) | after (s) | ratio |
| --- | --- | --- | --- |
| FPN |  |  |  |
| DeepLabV3 |  |  |  |
| LRASPP |  |  |  |
| Mask R-CNN |  |  |  |

# 3. ONNX Export Test

- 1차 지원 모델군

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
| YoloV5 | N | PTQ (Backbone only) | Y | Y (Backbone only) | 13 |

- 2차 지원 모델군

| Model | Pretrained | Quantization Support | FP32 model ONNX Export Support | Quantized model ONNX Export Support | ONNX opset |
| --- | --- | --- | --- | --- | --- |
| SqueezeNet 1.0 |  | PTQ, QAT | Y | Y | 13 |
| SqueezeNet 1.1 |  | PTQ, QAT | Y | Y | 13 |
| MNASNet_0,5 |  | PTQ, QAT | Y | Y | 13 |
| MNASNet_0,75 |  | PTQ, QAT | Y | Y | 13 |
| MNASNet_1.0 |  | PTQ, QAT | Y | Y | 13 |
| MNASNet_1.3 |  | PTQ, QAT | Y | Y | 13 |
| ConvNeXt |  | N | Y | N | 13 |
| RegNet X |  | PTQ, QAT | Y | Y | 13 |
| RegNet Y |  | N | Y | N | 13 |
| RetinaNet_ResNet50_FPN |  | PTQ, QAT | Y | N | 13 |
| RetinaNet_ResNet50_FPN_V2 |  | PTQ, QAT | Y | N | 13 |
| VGG16 |  | PTQ, QAT | Y | Y | 13 |
| VGG16_BN |  | PTQ, QAT | Y | Y | 13 |
| SSD300_VGG16 |  | PTQ, QAT | Y | N | 13 |
| SSDLite320_MobileNetV3_Large |  | PTQ, QAT | Y | N | 13 |
| FCOS |  | PTQ, QAT | Y | N | 13 |
| FPN |  |  |  |  |  |
| DeepLabV3 |  |  |  |  |  |
| LRASPP |  |  |  |  |  |
| Mask R-CNN |  |  |  |  |  |