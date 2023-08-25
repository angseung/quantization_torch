import onnx
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

from quantization_torch.utils.quantization_utils import cal_mse


class SingleChannelResNet(ResNet):
    """Modified ResNet to take just one channel as convolutional input"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Change to single channel for audio classification.
        self.conv1 = nn.Conv2d(
            1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )


def resnet18(**kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return SingleChannelResNet(BasicBlock, [2, 2, 2, 2], kwargs["num_classes"])


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        num_channels=32,
        rnn_type="lstm",
        rnn_hidden=64,
        dropout_p=0.2,
        batch_first=True,
        **kwargs,
    ):
        super(CRNN, self).__init__()
        input_size = kwargs.pop("input_size", None)
        batch_size = kwargs.pop("batch_size", None)

        data = torch.rand(batch_size, 1, input_size[0], input_size[1]).float()

        self.stem = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
        )
        data = self.stem(data)

        self.conv = nn.Sequential()
        for i in range(2):
            self.conv.add_module(
                "stage{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        num_channels,
                        num_channels * 2,
                        kernel_size=5,
                        stride=(2, 4),
                        padding=2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout_p),
                ),
            )
            num_channels *= 2

        data = self.conv(data)

        n, c, t, f = data.size()

        data = data.permute(0, 2, 3, 1)
        data = data.reshape(n, t, c * f)

        if rnn_type == "gru":
            rnn_cls = nn.GRU
        elif rnn_type == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError("rnn_type must be one of: lstm, gru")

        self.rnn = rnn_cls(
            input_size=c * f,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=batch_first,
        )

        data, h = self.rnn(data)
        data = data.reshape(n, -1)
        n, x = data.size()

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            # nn.BatchNorm1d(rnn_hidden * 2 * 16),
            nn.BatchNorm1d(x),
            # nn.Linear(rnn_hidden * 2 * 16, num_classes),
            nn.Linear(x, num_classes),
        )

    def forward(self, x):
        # conv features
        # print(x.shape)
        # x = x.unsqueeze(1)
        x = self.stem(x)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        # rnn features
        n, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1)
        # print(n)

        x = x.reshape(n, t, c * f)

        x, h = self.rnn(x)

        # fc features

        x = x.reshape(n, -1)

        x = self.fc(x)
        # print(x.shape)
        return x.squeeze()


class RNNBase(nn.Module):
    def __init__(
        self,
        rnn_type="lstm",
        rnn_hidden=64,
        batch_first=True,
        **kwargs,
    ):
        super(RNNBase, self).__init__()
        num_classes = kwargs["num_classes"]
        input_size = kwargs["input_size"]
        batch_size = kwargs["batch_size"]

        data = torch.rand(batch_size, 1, input_size).float()

        if rnn_type == "gru":
            rnn_cls = nn.GRU
        elif rnn_type == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError("rnn_type must be one of: lstm, gru")

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=batch_first,
        )

        data, h = self.rnn(data)

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(int(126 * rnn_hidden * 2 * 38.5), rnn_hidden * 2, bias=False),
            nn.Linear(data.size(2), rnn_hidden * 2, bias=False),
            nn.BatchNorm1d(rnn_hidden * 2),
            nn.Linear(rnn_hidden * 2, num_classes),
        )

    def forward(self, x):
        # rnn features
        try:
            x_temp = x.unsqueeze(1)
            x, h = self.rnn(x_temp)

        except:
            x, h = self.rnn(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc(x)
        return x


if __name__ == "__main__":
    # from torchvision.models.resnet import resnet18
    model = resnet18(num_classes=1000).eval()
    input_np = np.random.randn(1, 1, 224, 224).astype(np.float32)
    dummy_input = torch.from_numpy(input_np)
    dummy_output = model(dummy_input)
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        "../../onnx/a2ds_resnet18.onnx",
        do_constant_folding=True,
        verbose=True,
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
