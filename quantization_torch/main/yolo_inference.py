import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import yaml
from PIL import ImageFont, ImageDraw, Image


FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
fontpath = "fonts/NanumBarunGothic.ttf"
font = ImageFont.truetype(fontpath, 36)

from quantization_torch.models.common import DetectMultiBackend
from quantization_torch.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from quantization_torch.utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
)

from quantization_torch.utils.plots import Annotator, colors, save_one_box
from quantization_torch.utils.torch_utils import select_device, time_sync, normalizer, to_grayscale
from quantization_torch.utils.augment_utils import auto_canny
from quantization_torch.utils.quantization_utils import CalibrationDataLoader
from quantization_torch.models.yolo import yolo_model


@torch.no_grad()
def run(
    weights="weights/yolov5m-qat.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob, 0 for webcam
    imgsz=320,  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    normalize=True,
    gray=False,
    use_soft=False,
    edge=False,
    compile_model=False,
    quantize=False,
    nocal=False,
    show_best_epoch=False,
):
    assert not (
        normalize and gray
    )  # select gray or normalize. when selected both, escapes.

    # disable torch.compile if there is not any GPU
    if compile_model and not torch.cuda.is_available():
        compile_model = False

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load model
    device = select_device(device)

    if show_best_epoch and "yaml" not in str(weights):
        best_epoch = torch.load(weights, map_location=device)["epoch"]
        print(f"loading best scored model, {best_epoch}th...")

    model = DetectMultiBackend(weights, device=device, dnn=dnn)

    if compile_model:
        model.model = torch.compile(model.model)  # compile inference model

    stride, names, pt, jit, onnx = (
        model.stride,
        model.names,
        model.pt,
        model.jit,
        model.onnx,
    )
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    transform_normalize = normalizer()
    transform_to_gray = to_grayscale(num_output_channels=3)

    # Half
    half &= (
        pt and device.type != "cpu"
    )  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    if quantize:
        ## check yolo version
        if "v3" in weights:
            yolo_version = 3
        elif "v4" in weights:
            yolo_version = 4
        elif "v5" in weights:
            yolo_version = 5
        else:
            yolo_version = 5

        print(f"Detected Yolo Version is {yolo_version}.")

        if isinstance(model, DetectMultiBackend):
            model, head = yolo_model(
                model.model, quantize=True, is_qat=False, yolo_version=yolo_version
            )

        elif isinstance(model, torch.nn.Module):
            model, head = yolo_model(
                model, quantize=True, is_qat=False, yolo_version=yolo_version
            )

        if not nocal:
            # dataloader for calibration
            dataset = CalibrationDataLoader(os.path.join(ROOT, "data", "cropped"))
            calibration_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

            for i, img in enumerate(calibration_dataloader):
                print(f"\rcalibrating... {i + 1} / {dataset.__len__()}", end="")
                model(img)

        torch.ao.quantization.convert(model, inplace=True)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(
            source, img_size=imgsz, stride=stride, auto=pt and not jit
        )
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # warm-up GPU with temporary tensor if device is not CPU
    if pt and device.type != "cpu":
        model(
            torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters()))
        )
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        if edge:
            im = auto_canny(im.transpose([1, 2, 0]), return_rgb=True).transpose(
                [2, 0, 1]
            )

        im = torch.from_numpy(im.copy()).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        if normalize:
            im = transform_normalize(im)

        if gray:
            im = transform_to_gray(im)

        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = (
            increment_path(save_dir / Path(path).stem, mkdir=True)
            if visualize
            else False
        )
        if quantize:
            pred = head(model(im))
        else:
            pred = model(im, augment=augment, visualize=visualize)

        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes,
            agnostic_nms,
            max_det=max_det,
            use_soft=use_soft,
        )

        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # dump inference config to opt.yaml
        with open(save_dir / "opt.yaml", "w") as f:
            global opt
            opt_dict = vars(opt)
            opt_dict["weights"] = str(opt_dict["weights"])
            opt_dict["source"] = str(opt_dict["source"])

            if isinstance(opt_dict["imgsz"], list):
                opt_dict["imgsz"] = opt_dict["imgsz"][0]
            else:
                opt_dict["imgsz"] = str(opt_dict["imgsz"])

            opt_dict["project"] = str(opt_dict["project"])

            yaml.safe_dump(opt_dict, f, sort_keys=False)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        )  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(
                                xyxy,
                                imc,
                                file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                                BGR=True,
                            )

            # Print time (inference-only)
            LOGGER.info(f"{s}Done. ({t3 - t2:.3f}s)")

            # Stream results
            im0 = annotator.result()
            img_pillow = Image.fromarray(im0)
            draw = ImageDraw.Draw(img_pillow)
            im0 = np.array(img_pillow)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t
    )
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[320],
        help="inference size h,w",
    )
    parser.add_argument(
        "--compile-model", action="store_true", help="compile model (GPU ONLY)"
    )
    parser.add_argument(
        "--quantize", action="store_true", help="quantize model (CPU ONLY)"
    )
    parser.add_argument(
        "--nocal", action="store_true", help="skip calibration for quantization process"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="apply normalizer or not"
    )
    parser.add_argument("--gray", action="store_true", help="apply grayscale or not")
    parser.add_argument("--edge", action="store_true", help="apply canny edge or not")
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--use-soft", action="store_true", help="use soft nms rather than normal nms"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
