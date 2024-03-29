import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import *
from copy import deepcopy
import math
import platform
from pathlib import Path
from torchvision.models.quantization.utils import _fuse_modules
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch import nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from quantization_torch.models.common import *
from quantization_torch.models.experimental import *
from quantization_torch.utils.autoanchor import check_anchor_order
from quantization_torch.utils.general import LOGGER, check_version, make_divisible
from quantization_torch.utils.plots import feature_visualization
from quantization_torch.utils.torch_utils import (
    copy_attr,
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    scale_img,
    time_sync,
)
from quantization_torch.utils.quantization_utils import (
    CalibrationDataLoader,
    get_platform_aware_qconfig,
    cal_mse,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.nl, -1, 2)
        )  # shape(nl,na,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[
                        i
                    ]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for  on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(
            torch.__version__, "1.10.0"
        ):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(
                [torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing="ij"
            )
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            (self.anchors[i].clone() * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand((1, self.na, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(
        self, cfg="yolov3.yaml", ch=3, nc=None, anchors=None
    ):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch]
        )  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(
            x, profile, visualize
        )  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip  augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = (
            thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2
            if thop
            else 0
        )  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(
                f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}"
            )
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.999999))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ("%6g Conv2d.bias:" + "%10.3g" * 6)
                % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info("Adding AutoShape... ")
        m = AutoShape(self)  # wrap model
        copy_attr(
            m, self, include=("yaml", "nc", "hyp", "names", "stride"), exclude=()
        )  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
    )
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            Conv,
            ConvBnReLU,
            GhostConv,
            Bottleneck,
            BottleneckReLU,
            DWSBottleneck,
            DWSBottleneckReLU,
            GhostBottleneck,
            SPP,
            SPPReLU,
            SPPF,
            SPPFReLU,
            DWConv,
            DWConvReLU,
            DWSConvReLU,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            BottleneckCSPReLU,
            C3,
            C3ReLU,
            C3TR,
            C3SPP,
            C3Ghost,
        ]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, BottleneckCSPReLU, C3, C3TR, C3Ghost, C3ReLU]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is DWSConv:
            c2 = args[1]
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        )  # module

        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        LOGGER.info(
            f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}"
        )  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []  # a list to save a dimension output channels of the previous layer
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class QuantizableYoloBackbone(nn.Module):
    def __init__(self, model: Union[str, nn.Module] = None, yolo_version: int = 3):
        super().__init__()
        self.yolo_version = yolo_version

        if isinstance(model, str):
            if model.endswith(".pt"):
                self.model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
                if isinstance(self.model, dict):
                    self.model = self.model["model"].float()
            elif model.endswith(".yaml"):
                self.model = DetectMultiBackend(
                    os.path.join(ROOT, model), torch.device("cpu"), False
                ).model
        elif isinstance(model, nn.Module):
            self.model = model

        elif isinstance(model, DetectMultiBackend):
            self.model = model.model

        else:
            raise AttributeError("Unsupported model type")

        self.quant = torch.ao.quantization.QuantStub()

        self.model.model[-1] = nn.Identity()

        self.dequant = torch.ao.quantization.DeQuantStub()

    def fuse_model(self, is_qat: bool = False) -> None:
        fuse_yolo(self.model, is_qat=is_qat)

    def _forward_impl_v3(self, x: Tensor) -> List[Tensor]:
        x = self.quant(x)

        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "18":
                    x = block([x17, x8])
                elif i == "25":
                    x = block([x24, x6])
            else:  # ConvBnReLU, BottleneckReLU
                if i == "16":
                    x = block(x14)
                elif i == "23":
                    x = block(x21)
                else:
                    x = block(x)

                # save feature map for concat/conv layers...
                if i == "6":
                    x6 = x.clone()
                elif i == "8":
                    x8 = x.clone()
                elif i == "14":
                    x14 = x.clone()
                elif i == "15":
                    x15 = x.clone()
                elif i == "17":
                    x17 = x.clone()
                elif i == "21":
                    x21 = x.clone()
                elif i == "22":
                    x22 = x.clone()
                elif i == "24":
                    x24 = x.clone()
                elif i == "27":
                    x27 = x.clone()

        x15 = self.dequant(x15)
        x22 = self.dequant(x22)
        x27 = self.dequant(x27)

        return [x27, x22, x15]

    def _forward_impl_v4(self, x: Tensor) -> List[Tensor]:
        x = self.quant(x)

        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "25":
                    x = block([x24, x12])
                elif i == "29":
                    x = block([x28, x9])
                elif i == "32":
                    x = block([x31, x27])
                elif i == "35":
                    x = block([x34, x23])
            else:  # ConvBnReLU, BottleneckReLU, BottleneckCSPReLU, SPPReLU
                x = block(x)

                # save feature map for concat/conv layers...
                if i == "9":
                    x9 = x.clone()
                elif i == "12":
                    x12 = x.clone()
                elif i == "23":
                    x23 = x.clone()
                elif i == "24":
                    x24 = x.clone()
                elif i == "27":
                    x27 = x.clone()
                elif i == "28":
                    x28 = x.clone()
                elif i == "30":
                    x30 = x.clone()
                elif i == "31":
                    x31 = x.clone()
                elif i == "33":
                    x33 = x.clone()
                elif i == "34":
                    x34 = x.clone()
                elif i == "36":
                    x36 = x.clone()

        x30 = self.dequant(x30)
        x33 = self.dequant(x33)
        x36 = self.dequant(x36)

        return [x30, x33, x36]

    def _forward_impl_v5(self, x: Tensor) -> List[Tensor]:
        x = self.quant(x)

        for i, block in self.model.model.named_children():
            if isinstance(block, Concat):
                if i == "12":
                    x = block([x11, x6])
                elif i == "16":
                    x = block([x15, x4])
                elif i == "19":
                    x = block([x18, x14])
                elif i == "22":
                    x = block([x21, x10])
            else:  # ConvBnReLU, BottleneckReLU, C3ReLU, SPPFReLU
                x = block(x)

                # save feature map for concat/conv layers...
                if i == "4":
                    x4 = x.clone()
                elif i == "6":
                    x6 = x.clone()
                elif i == "10":
                    x10 = x.clone()
                elif i == "11":
                    x11 = x.clone()
                elif i == "14":
                    x14 = x.clone()
                elif i == "15":
                    x15 = x.clone()
                elif i == "17":
                    x17 = x.clone()
                elif i == "18":
                    x18 = x.clone()
                elif i == "20":
                    x20 = x.clone()
                elif i == "21":
                    x21 = x.clone()
                elif i == "23":
                    x23 = x.clone()

        x17 = self.dequant(x17)
        x20 = self.dequant(x20)
        x23 = self.dequant(x23)

        return [x17, x20, x23]

    def forward(
        self, x: Union[Tensor, List[Tensor]]
    ) -> Union[Tuple[List[Tensor], Tensor], List[Tensor]]:
        if self.yolo_version == 3:
            return self._forward_impl_v3(x)

        elif self.yolo_version == 4:
            return self._forward_impl_v4(x)

        elif self.yolo_version == 5:
            return self._forward_impl_v5(x)


class YoloHead(nn.Module):
    def __init__(self, model: Union[str, nn.Module] = None):
        super().__init__()
        if isinstance(model, str):
            if model.endswith(".pt"):
                yolo_model = torch.load(
                    os.path.join(ROOT, model), map_location=torch.device("cpu")
                )
                if isinstance(yolo_model, dict):
                    yolo_model = yolo_model["model"].float()
            elif model.endswith(".yaml"):
                yolo_model = DetectMultiBackend(
                    os.path.join(ROOT, model), torch.device("cpu"), False
                ).model
        elif isinstance(model, nn.Module):
            yolo_model = model

        elif isinstance(model, DetectMultiBackend):
            yolo_model = model.model

        else:
            raise AttributeError("Unsupported model type")

        self.model = deepcopy(yolo_model.model[-1])

        self.model.eval()

    def forward(
        self, x: List[Tensor]
    ) -> Union[Tuple[List[Tensor], Tensor], List[Tensor]]:
        return self.model(x) if self.model.training else self.model(x)[0]


def fuse_yolo(blocks: nn.Module, is_qat: bool = False):
    """
    A function for fusing conv-bn-relu layers
    Parameters
    ----------
    blocks: A nn.Module type model to be fused
    is_qat
    -------

    """
    for _, block in blocks.named_children():
        if isinstance(block, ConvBnReLU):
            _fuse_modules(block, [["conv", "bn", "act"]], is_qat=is_qat, inplace=True)
        else:
            fuse_yolo(block, is_qat=is_qat)


def yolo_model(
    model: Union[str, nn.Module],
    yolo_version: int = 3,
    quantize: bool = False,
    is_qat: bool = False,
) -> Tuple[QuantizableYoloBackbone, YoloHead]:
    yolo_head = YoloHead(model)
    yolo_backbone = QuantizableYoloBackbone(model, yolo_version=yolo_version)

    backend = get_platform_aware_qconfig()
    if backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    if quantize:
        if is_qat:
            yolo_backbone.fuse_model(is_qat=True)
            yolo_backbone.qconfig = torch.ao.quantization.get_default_qat_qconfig(
                backend
            )
            yolo_backbone.train()
            torch.ao.quantization.prepare_qat(yolo_backbone, inplace=True)
        else:
            yolo_backbone.fuse_model(is_qat=False)
            yolo_backbone.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            torch.ao.quantization.prepare(yolo_backbone, inplace=True)

    return yolo_backbone, yolo_head


if __name__ == "__main__":
    # create a model instance
    architecture = platform.uname().machine
    dataset = CalibrationDataLoader(os.path.join(ROOT, "data", "cropped"))
    calibration_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input = torch.randn(1, 3, 320, 320)
    fname = os.path.join("models", "yolov5x-qat.yaml")
    yolo_qint8, yolo_detector = yolo_model(fname, yolo_version=5)
    yolo_fp32 = QuantizableYoloBackbone(fname, yolo_version=5)

    for i, img in enumerate(calibration_dataloader):
        print(f"\rcalibrating... {i + 1} / {dataset.__len__()}", end="")
        yolo_qint8(img)

    torch.ao.quantization.convert(yolo_qint8, inplace=True)
    dummy_output = yolo_qint8(input)
    pred = yolo_detector(dummy_output)
    pred_fp32 = yolo_detector(yolo_fp32(input))
    nmse = cal_mse(pred, pred_fp32, norm=True)
