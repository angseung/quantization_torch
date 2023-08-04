import os
import time
from pathlib import Path
from typing import Union, Tuple, List, Optional, Dict, Type
import cv2
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from quantization_torch.utils.hyps import *
from quantization_torch.utils.torch_utils import normalizer
from quantization_torch.utils.general import non_max_suppression

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def parse_detection_results(fname: str) -> np.ndarray:
    """
    parses the label file, then converts it to np.ndarray type
    Args:
        fname: detection_results file name

    Returns: label in yolo format as np.ndarray

    """
    with open(fname, encoding="utf-8") as f:
        bboxes = f.readlines()
        label = []

    for bbox in bboxes:
        label.append(bbox.split()[:-1])

    return np.array(label, dtype=np.float64)


def unsharp(img: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    blr = cv2.GaussianBlur(img, (0, 0), 2)
    dst = np.clip((1 + alpha) * img - alpha * blr, 0, 255).astype(np.uint8)

    return dst


def resize(img: np.ndarray, size: Union[Tuple[int, int], int]) -> np.ndarray:
    is_upsample = False

    if len(img.shape) == 2:
        height, width = img.shape
    elif len(img.shape) == 3:
        height, width, channel = img.shape

    if isinstance(size, int):
        ratio = height / width if height > width else width / height
        dsize = (
            (size, int(size / ratio)) if width > height else (int(size / ratio), size)
        )

    elif isinstance(size, tuple):
        if len(size) != 2:
            raise ValueError
        dsize = size

    # check whether resize operation is upsampling or downsampling
    if dsize[0] * dsize[1] > height * width:
        is_upsample = True

    return (
        cv2.resize(
            img, dsize=dsize, interpolation=cv2.INTER_AREA
        )  # use inter_area interpolation method when downsample image.
        if not is_upsample
        else cv2.resize(
            img, dsize=dsize, interpolation=cv2.INTER_LINEAR
        )  # use inder_linear interpolation method when upsample image.
    )


def get_binarized_img(
    img: np.ndarray, blur_opt: bool = True, blur_kernal_size: Optional[int] = 5
) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Morphological Transformation (https://dsbook.tistory.com/203)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return (
        cv2.GaussianBlur(gray, ksize=(blur_kernal_size, blur_kernal_size), sigmaX=0)
        if blur_opt
        else gray
    )


def get_thresh_img(img: np.ndarray, mode: Optional[Union[int, str]] = 1) -> np.ndarray:
    if mode not in [0, 1, "normal", "adaptive"]:
        raise ValueError

    if mode in ["normal", 0]:
        # mode 1. normal threshold
        # ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ret, binary = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
        img_thresh = cv2.bitwise_not(binary)

    elif mode in ["adaptive", 1]:
        # mode 2. adaptive threshold
        img_thresh = cv2.adaptiveThreshold(
            img,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9,
        )

    return img_thresh


def get_black_and_white_img(img: np.ndarray, output_inverse: bool = True) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 7)
    img = cv2.Laplacian(img, cv2.CV_8U, 5)
    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_not(img) if output_inverse else img


def find_roi(img: np.ndarray, img_thresh: np.ndarray) -> List[Dict[str, int]]:
    assert img.shape[:-1] == img_thresh.shape
    contours, _ = cv2.findContours(
        img_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = img_thresh.shape[:2]
    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_dict.append(
            {
                "contour": contour,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + (w / 2),
                "cy": y + (h / 2),
            }
        )

    possible_contours = []
    cnt = 0

    for d in contours_dict:
        area = d["w"] * d["h"]
        ratio = d["w"] / d["h"]

        if (
            area > MIN_AREA
            and MIN_WIDTH < d["w"] < MAX_WIDTH
            and MIN_HEIGHT < d["h"] < MAX_HEIGHT
            and MIN_RATIO < ratio < MAX_RATIO
        ):
            d["idx"] = cnt
            cnt += 1
            possible_contours.append(d)

    def find_chars(contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1["idx"] == d2["idx"]:
                    continue

                dx = abs(d1["cx"] - d2["cx"])
                dy = abs(d1["cy"] - d2["cy"])

                diagonal_length1 = np.sqrt(d1["w"] ** 2 + d1["h"] ** 2)

                distance = np.linalg.norm(
                    np.array([d1["cx"], d1["cy"]]) - np.array([d2["cx"], d2["cy"]])
                )
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))

                area_diff = abs(d1["w"] * d1["h"] - d2["w"] * d2["h"]) / (
                    d1["w"] * d1["h"]
                )

                width_diff = abs(d1["w"] - d2["w"]) / d1["w"]
                height_diff = abs(d1["h"] - d2["h"]) / d1["h"]

                if (
                    distance < diagonal_length1 * MAX_DIAG_MULTIPLIER
                    and angle_diff < MAX_ANGLE_DIFF
                    and area_diff < MAX_AREA_DIFF
                    and width_diff < MAX_WIDTH_DIFF
                    and height_diff < MAX_HEIGHT_DIFF
                ):
                    matched_contours_idx.append(d2["idx"])

            matched_contours_idx.append(d1["idx"])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            # find matched contour for remainder contours
            unmatched_contour_idx = []

            for d4 in contour_list:
                if d4["idx"] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4["idx"])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)
    matched_result = []

    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    plate_infos = contour_to_plate_region_rev(matched_result)

    return plate_infos


def convert_contour(
    contours: List[Dict],
    imgsz: Tuple[int, int],
    target_imgsz: Tuple[int, int],
) -> List[np.ndarray]:
    """
    It converts contours size to original shape
    :param contours: contours to be resized
    :param imgsz: image size before resize
    :param target_imgsz: resired image size
    :return: scaled contours
    """

    ratio_height, ratio_width = (
        target_imgsz[1] / imgsz[1]
        if target_imgsz[1] > imgsz[1]
        else imgsz[1] / target_imgsz[1],
        target_imgsz[0] / imgsz[0]
        if target_imgsz[0] > imgsz[0]
        else imgsz[0] / target_imgsz[0],
    )

    for contour in contours:
        contour["x"] = int(contour["x"] * ratio_width)
        contour["y"] = int(contour["y"] * ratio_height)
        contour["w"] = int(contour["w"] * ratio_width)
        contour["h"] = int(contour["h"] * ratio_height)

    return contours


def show_contours(
    img: np.ndarray, result: List, return_img: bool = False
) -> Union[None, np.ndarray]:
    temp_result = img.copy()

    for r in result:
        for d in r:
            cv2.rectangle(
                temp_result,
                pt1=(d["x"], d["y"]),
                pt2=(d["x"] + d["w"], d["y"] + d["h"]),
                color=(0, 0, 255),
                thickness=2,
            )
    plt.figure()
    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap="gray")
    plt.show()
    plt.savefig(f"../c_outputs/{fname}")

    if return_img:
        return temp_result


def clip(val: int, lower: int = 0, higher: int = 255) -> int:
    if lower <= val <= higher:
        return val
    elif val < lower:
        return lower
    else:
        return higher


def contour_to_plate_region(
    matched_result: List[np.ndarray], char_size: int = 25
) -> Tuple[List, float]:
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x["cx"])
        char_height_list = [sorted_chars[i]["h"] for i in range(len(sorted_chars))]
        char_height_mean = np.mean(char_height_list)
        crop_ratio = char_size / char_height_mean

        plate_cx = (sorted_chars[0]["cx"] + sorted_chars[-1]["cx"]) / 2
        plate_cy = (sorted_chars[0]["cy"] + sorted_chars[-1]["cy"]) / 2

        plate_width = (
            sorted_chars[-1]["x"] + sorted_chars[-1]["w"] - sorted_chars[0]["x"]
        )

        sum_height = 0

        for d in sorted_chars:
            sum_height += d["h"]

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        plate_infos.append(
            {
                "x": int(plate_cx - plate_width / 2),
                "y": int(plate_cy - plate_height / 2),
                "w": int(plate_width),
                "h": int(plate_height),
            }
        )

        break

    return plate_infos, crop_ratio


def contour_to_plate_region_rev(matched_result: List[np.ndarray]) -> List:
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x["cx"])

        plate_cx = (sorted_chars[0]["cx"] + sorted_chars[-1]["cx"]) / 2
        plate_cy = (sorted_chars[0]["cy"] + sorted_chars[-1]["cy"]) / 2

        plate_width = (
            sorted_chars[-1]["x"] + sorted_chars[-1]["w"] - sorted_chars[0]["x"]
        ) * PLATE_WIDTH_PADDING

        sum_height = 0

        for d in sorted_chars:
            sum_height += d["h"]

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        plate_infos.append(
            {
                "x": int(plate_cx - plate_width / 2),
                "y": int(plate_cy - plate_height / 2),
                "w": int(plate_width),
                "h": int(plate_height),
            }
        )

    return plate_infos


def detect_roi_with_yolo(
    img: np.ndarray,
    model: nn.Module,
    iou_thres: float = 0.1,
    conf_thres: float = 0.25,
    device: Type[torch.device] = torch.device("cpu"),
    normalize: bool = True,
) -> Union[List[Dict[str, Union[int, float]]], None]:
    # convert image format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB
    img = img.transpose([2, 0, 1])  # convert [H, W, C] to [C, H, W]

    # convert np.ndarray to torch.tensor
    img_tensor = torch.from_numpy(img).float()  # uint8 to fp16/32
    img_tensor /= 255  # 0 - 255 to 0.0 - 1.0

    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]  # expand for batch dim

    # normalize input image
    if normalize:
        img_tensor = normalizer()(img_tensor)

    # inference with yolo model
    pred = model(img_tensor, augment=False, visualize=False, val=False)

    # nms
    # pred can be mapped with matched_results in a cv-based algorithm.
    pred = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        agnostic=False,
        max_det=1000,
        use_soft=False,
    )

    # convert bboxes to list of Dict[str, int] format
    contours: List[Dict[str, Union[int, float]]] = []
    pred_np = pred[0].numpy()[:, :4]

    if pred_np.size:
        for bbox in pred_np:
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2] - bbox[0])
            h = int(bbox[3] - bbox[1])
            cx = x + (w / 2)
            cy = y + (h / 2)

            contours.append(
                {
                    "contour": None,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cx": cx,
                    "cy": cy,
                }
            )

        matched_result = [np.array(contours)]

        # convert contours of chars to contours of region of plate
        contours, crop_ratio = contour_to_plate_region(matched_result, char_size=25)

        return contours

    else:
        return None


def crop_region_of_plates(
    img: np.ndarray,
    model: Union[nn.Module, None],
    target_imgsz: int = 320,
    imgsz: Union[int, None, List[int]] = 640,
    char_size: int = 25,
    use_yolo: bool = False,
    top_only: bool = True,
    img_show_opt: bool = False,
    return_as_img: bool = False,
) -> Union[np.ndarray, Tuple[int, int, int, int]]:
    img_ori = img.copy()  # BGR
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    height_ori, width_ori = img_ori.shape[:2]

    # resize input image
    if imgsz is not None:
        if isinstance(imgsz, list):
            img = resize(img_ori, imgsz[0])

        else:
            img = resize(img_ori, imgsz)

    height, width = img.shape[:2]

    if use_yolo:  # use ml-based algorithm
        contours: List[np.ndarray] = detect_roi_with_yolo(img, model)

    else:  # use cv-based algorithm
        # convert img to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # blur image to reduce noise
        blurred = cv2.GaussianBlur(img_gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

        # detect edge with canny edge detection
        img1 = cv2.Canny(blurred, CANNY_LOWER_THRESH, CANNY_UPPER_THRESH)

        if img_show_opt:
            plt.imshow(img1)
            plt.show()

        contours: List[np.ndarray] = find_roi(img, img1)

    if contours:
        # rescale bboxes to the original shape
        contours = convert_contour(
            contours,
            imgsz=(width, height),
            target_imgsz=(width_ori, height_ori),
        )

        for contour in contours:
            # TODO: Implement crop size decision codes based the size of each chars in plates...
            plate_width = int(contour["w"] * PLATE_WIDTH_PADDING)
            plate_height = int(contour["h"] * PLATE_HEIGHT_PADDING)

            padding_left_right = (plate_width - contour["w"]) // 2
            padding_upper_lower = (plate_height - contour["h"]) // 2

            xtl_cropped = clip(
                contour["x"] - padding_left_right, lower=0, higher=width_ori
            )
            ytl_cropped = clip(
                contour["y"] - padding_upper_lower, lower=0, higher=height_ori
            )

            xbr_cropped = clip(
                xtl_cropped + padding_left_right + plate_width,
                lower=0,
                higher=width_ori,
            )
            ybr_cropped = clip(
                ytl_cropped + padding_upper_lower + plate_height,
                lower=0,
                higher=width_ori,
            )

            img_cropped = img_ori[
                ytl_cropped:ybr_cropped,
                xtl_cropped:xbr_cropped,
                ...,
            ]  # (H, W, C)

            if img_show_opt:
                fig = plt.figure()
                plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
                plt.show()

            if top_only:
                break

        return (
            resize(img_cropped, target_imgsz)
            if return_as_img
            else (xtl_cropped, ytl_cropped, xbr_cropped, ybr_cropped)
        )

    # return cropped image if return_as_img is True
    # else return Tuple: (xtl, ytl, xbr, ybr)
    else:
        img_ori = resize(img_ori, target_imgsz)
        return img_ori if return_as_img else (0, 0, img_ori.shape[1], img_ori.shape[0])


def rescale_roi(
    region_of_roi: Tuple[int, int, int, int],
    prev_shape: Tuple[int, int],
    resized_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    ratio_of_width = prev_shape[0] / resized_shape[0]
    ratio_of_height = prev_shape[1] / resized_shape[1]

    region_of_roi_scaled = (
        int(region_of_roi[0] * ratio_of_width),
        int(region_of_roi[1] * ratio_of_height),
        int(region_of_roi[2] * ratio_of_width),
        int(region_of_roi[3] * ratio_of_height),
    )

    return region_of_roi_scaled


if __name__ == "__main__":
    img_dir = "../regions"
    target_dir = "../../outputs"
    img_list = os.listdir(img_dir)

    for fname in img_list:
        if fname[0] == ".":
            continue

        start = time.time()
        img = cv2.imread(f"{img_dir}/{fname}")
        img_cropped = crop_region_of_plates(
            img=img,
            target_imgsz=320,
            imgsz=640,
            top_only=True,
            img_show_opt=False,
        )
        cv2.imwrite(f"{target_dir}/{fname}", img_cropped)
