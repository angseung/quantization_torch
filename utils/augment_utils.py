from typing import Tuple, Union, Optional
import random
from datetime import datetime
from PIL import Image, ImageDraw
import numpy as np
import cv2
from matplotlib import pyplot as plt


def parse_label(fname: str) -> np.ndarray:
    """
    parses the label file, then converts it to np.ndarray type
    Args:
        fname: label file name

    Returns: label as np.ndarray

    """
    with open(fname, encoding="utf-8") as f:
        bboxes = f.readlines()
        label = []

    for bbox in bboxes:
        label.append(bbox.split())

    return np.array(label, dtype=np.float64)


def label_yolo2voc(label_yolo: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    converts label format from yolo to voc
    Args:
        label_yolo: (x_center, y_center, w, h), normalized
        h: img height
        w: img width

    Returns: (xtl, ytl, xbr, ybr)

    """
    label_voc = np.zeros(label_yolo.shape, dtype=np.float64)
    label_voc[:, 0] = label_yolo[:, 0]

    label_yolo_temp = label_yolo.copy()
    label_yolo_temp[:, [1, 3]] *= w
    label_yolo_temp[:, [2, 4]] *= h

    # convert x_center, y_center to xtl, ytl
    label_voc[:, 1] = label_yolo_temp[:, 1] - 0.5 * label_yolo_temp[:, 3]
    label_voc[:, 2] = label_yolo_temp[:, 2] - 0.5 * label_yolo_temp[:, 4]

    # convert width, height to xbr, ybr
    label_voc[:, 3] = label_voc[:, 1] + label_yolo_temp[:, 3]
    label_voc[:, 4] = label_voc[:, 2] + label_yolo_temp[:, 4]

    return label_voc.astype(np.uint32)


def label_voc2yolo(label_voc: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    converts label format from voc to yolo
    Args:
        label_voc: (xtl, ytl, xbr, ybr)
        h: img heights
        w: img width

    Returns: (x_center, y_center, w, h), normalized

    """
    label_yolo = np.zeros(label_voc.shape, dtype=np.float64)
    label_yolo[:, 0] = label_voc[:, 0]

    # convert xtl, ytl to x_center, y_center
    label_yolo[:, 1] = 0.5 * (label_voc[:, 1] + label_voc[:, 3])
    label_yolo[:, 2] = 0.5 * (label_voc[:, 2] + label_voc[:, 4])

    # convert xbr, ybr to width, height
    label_yolo[:, 3] = label_voc[:, 3] - label_voc[:, 1]
    label_yolo[:, 4] = label_voc[:, 4] - label_voc[:, 2]

    # normalize
    label_yolo[:, [1, 3]] /= w
    label_yolo[:, [2, 4]] /= h

    return label_yolo


def find_draw_region(
    img: np.ndarray, label: np.ndarray, foreground: np.ndarray
) -> Tuple[int, int, int, int, int]:
    """
    find region for drawing foreground images
    Args:
        img: background image
        label: annotation label of img
        foreground: foreground image to be appended on background image

    Returns: selected region to be appended

    """
    h, w = img.shape[:2]
    h_fg, w_fg = foreground.shape[:2]
    label_pixel = np.copy(label)
    label_pixel[:, [1, 3]] *= w
    label_pixel[:, [2, 4]] *= h

    # convert x_center, y_center to xtl, ytl
    label_pixel[:, 1] = label_pixel[:, 1] - 0.5 * label_pixel[:, 3]
    label_pixel[:, 2] = label_pixel[:, 2] - 0.5 * label_pixel[:, 4]

    # convert width, height to xbr, ybr
    label_pixel[:, 3] = label_pixel[:, 1] + label_pixel[:, 3]
    label_pixel[:, 4] = label_pixel[:, 2] + label_pixel[:, 4]

    label_pixel = label_pixel.astype(np.uint32)

    xtl = label_pixel[:, 1].min()
    ytl = label_pixel[:, 2].min()
    xbr = label_pixel[:, 3].max()
    ybr = label_pixel[:, 4].max()

    region_candidate = [False] * 8
    region_area = [0.0] * 8

    # region 1
    w1, h1 = xtl, ytl
    region_candidate[0] = (w1 >= w_fg) and (h1 >= h_fg)
    region_area[0] = w1 * h1

    # region 2
    w2, h2 = xbr - xtl, ytl
    region_candidate[1] = (w2 >= w_fg) and (h2 >= h_fg)
    region_area[1] = w2 * h2

    # region 3
    w3, h3 = w - xbr, ytl
    region_candidate[2] = (w3 >= w_fg) and (h3 >= h_fg)
    region_area[2] = w3 * h3

    # region 4
    w4, h4 = xtl, ybr - ytl
    region_candidate[3] = (w4 >= w_fg) and (h4 >= h_fg)
    region_area[3] = w4 * h4

    # region 5
    w5, h5 = w - xbr, ybr - ytl
    region_candidate[4] = (w5 >= w_fg) and (h5 >= h_fg)
    region_area[4] = w5 * h5

    # region 6
    w6, h6 = xtl, h - ybr
    region_candidate[5] = (w6 >= w_fg) and (h6 >= h_fg)
    region_area[5] = w6 * h6

    # region 7
    w7, h7 = xbr - xtl, h - ybr
    region_candidate[6] = (w7 >= w_fg) and (h7 >= h_fg)
    region_area[6] = w7 * h7

    # region 8
    w8, h8 = w - xbr, h - ybr
    region_candidate[7] = (w8 >= w_fg) and (h8 >= h_fg)
    region_area[7] = w8 * h8

    region_candidate = np.array(region_candidate, dtype=np.uint32)
    region_area = np.array(region_area, dtype=np.uint32)

    # no selected region
    if (region_candidate * region_area).sum() < 0.5:
        selected_region = 0
    else:
        selected_region = (region_candidate * region_area).argmax() + 1

    if selected_region == 1:
        area = (selected_region, 0, 0, xtl.item(), ytl.item())
    elif selected_region == 2:
        area = (selected_region, xtl.item(), 0, xbr.item(), ytl.item())
    elif selected_region == 3:
        area = (selected_region, xbr.item(), 0, w, ytl.item())
    elif selected_region == 4:
        area = (selected_region, 0, ytl.item(), xtl.item(), ybr.item())
    elif selected_region == 5:
        area = (selected_region, xbr.item(), ytl.item(), w, ybr.item())
    elif selected_region == 6:
        area = (selected_region, 0, ybr.item(), xtl.item(), h)
    elif selected_region == 7:
        area = (selected_region, xtl.item(), ybr.item(), xbr.item(), h)
    elif selected_region == 8:
        area = (selected_region, xbr.item(), ybr.item(), w, h)
    elif selected_region == 0:
        area = (selected_region, 0, 0, 0, 0)

    return area


def write_label(target_dir: str, fname: str, bboxes: np.ndarray) -> None:
    """
    exports np.ndarray label to txt file
    Args:
        target_dir: save dir for label file
        fname: file name of label
        bboxes: annotation information, np.ndarray type

    Returns: None

    """
    num_boxes = bboxes.shape[0]

    with open(f"{target_dir}/{fname}.txt", "w") as f:
        for i in range(num_boxes):
            target_str = f"{int(bboxes[i][0])} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]} {bboxes[i][4]}"
            f.write(f"{target_str}\n")


def draw_bbox_on_img(
    img: np.ndarray,
    label: Union[np.ndarray, str],
    color: Union[Tuple[int, int, int], str],
    transpose: Optional[bool] = False,
) -> np.ndarray:
    if isinstance(label, str):
        label = parse_label(label)  # label must be np.ndarray

    # if label is yolo format
    if label[0, 0] < 1.0:
        label = label_yolo2voc(label, *(img.shape[:2]))

    if transpose:
        img = img.transpose(1, 0, 2)
        label_tr = label.copy()
        label_tr[:, 1], label_tr[:, 2] = label[:, 2], label[:, 1]
        label_tr[:, 3], label_tr[:, 4] = label[:, 4], label[:, 3]
        label = label_tr.copy()

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i in range(label.shape[0]):
        pos = tuple(label[i][1:].tolist())
        draw.rectangle(pos, outline=color, width=3)

    return np.asarray(img)


def augment_img(
    fg_img: np.ndarray, fg_label: np.ndarray, bg_img: np.ndarray, bg_label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    if len(fg_img.shape) != 2:
        if fg_img.shape[2] == 4:
            mode = "blend"
        else:
            mode = "override"
    else:
        mode = "blend"

    bg_h, bg_w = bg_img.shape[:2]
    fg_h, fg_w = fg_img.shape[:2]

    selected_region, area_xtl, area_ytl, area_xbr, area_ybr = find_draw_region(
        bg_img, bg_label, fg_img
    )
    if not (area_xbr > area_xtl and area_ybr > area_ytl):
        return bg_img, bg_label, False

    allowed_width = area_xbr - area_xtl - fg_w
    allowed_height = area_ybr - area_ytl - fg_h

    # get draw point offset
    try:
        draw_xtl = random.randint(0, allowed_width - 1)
        draw_ytl = random.randint(0, allowed_height - 1)

    # do nothing if failed
    except:
        return bg_img, bg_label, False

    abs_xtl, abs_ytl = draw_xtl + area_xtl, draw_ytl + area_ytl

    # draw fg_img on bg_img
    if mode == "override":
        bg_img[abs_ytl : abs_ytl + fg_h, abs_xtl : abs_xtl + fg_w, :] = fg_img
    elif mode == "blend":
        blended = blend_bgra_on_bgr(fg=fg_img, bg=bg_img, row=abs_ytl, col=abs_xtl)
        bg_img[abs_ytl : abs_ytl + fg_h, abs_xtl : abs_xtl + fg_w, :] = blended

    # compensate bbox offset of fg_label
    fg_label_voc = label_yolo2voc(fg_label, fg_h, fg_w)
    fg_label_voc[:, [1, 3]] += abs_xtl
    fg_label_voc[:, [2, 4]] += abs_ytl
    fg_label_yolo = label_voc2yolo(fg_label_voc, bg_h, bg_w)

    label = np.concatenate((fg_label_yolo, bg_label), axis=0)

    return bg_img, label, True


def random_resize(
    img: np.ndarray,
    label: Optional[Union[np.ndarray, None]] = None,
    scale_min: Union[int, float] = 0.75,
    scale_max: Union[int, float] = 2.5,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    random.seed(datetime.now().timestamp())
    scaled = random.uniform(scale_min, scale_max)
    h, w = img.shape[:2]

    if h > w:
        ratio = h / w
        w_scaled = w * scaled
        h_scaled = w_scaled * ratio

    else:
        ratio = w / h
        h_scaled = h * scaled
        w_scaled = h_scaled * ratio

    size = int(w_scaled), int(h_scaled)

    if label is not None:
        label = label_yolo2voc(label, h, w).astype(np.float64)
        label[:, 1:] *= scaled
        label = label_voc2yolo(label, h_scaled, w_scaled)

        return cv2.resize(img, size, interpolation=cv2.INTER_AREA), label

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def blend_bgra_on_bgr(fg: np.ndarray, bg: np.ndarray, row: int, col: int) -> np.ndarray:
    _, mask = cv2.threshold(fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.cvtColor(fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = bg[row : row + h, col : col + w]

    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    blended = masked_fg + masked_bg

    return blended


def blend_bgra_on_bgra(
    fg: np.ndarray, bg: np.ndarray, row: int, col: int
) -> np.ndarray:
    assert fg.shape[2] == 4 and bg.shape[2] == 4

    padded_fg = np.zeros_like(bg, dtype=np.uint8)
    h, w = fg.shape[:2]
    padded_fg[row : row + h, col : col + w, :] = fg

    _, mask_fg = cv2.threshold(bg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    _, mask_bg = cv2.threshold(padded_fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    alpha = cv2.bitwise_or(mask_fg, mask_bg)

    bg[:, :, :3] = blend_bgra_on_bgr(bg=bg[:, :, :3], fg=padded_fg, row=0, col=0)

    blue, green, red = cv2.split(bg[:, :, :3])
    bgra = [blue, green, red, alpha]

    return cv2.merge(bgra)


def blend_bgr_on_bgra(fg: np.ndarray, bg: np.ndarray, row: int, col: int) -> np.ndarray:
    assert fg.shape[2] == 3 and bg.shape[2] == 4
    h, w = fg.shape[:2]
    bg[row : row + h, col : col + w, :3] = fg

    return bg


def auto_canny(
    image: np.ndarray, sigma: float = 0.33, return_rgb: bool = False
) -> np.ndarray:
    image = cv2.GaussianBlur(image, (7, 7), 0)
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    if return_rgb:
        edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)

    # return the edged image
    return edged


if __name__ == "__main__":
    base_dir = "../data/thermal2"
    fname = "frame_00185_1"
    label = parse_label(f"{base_dir}/labels/train/{fname}.txt")
    img = cv2.imread(f"{base_dir}/images/train/{fname}.jpg")

    img = draw_bbox_on_img(img, label, color=(255, 255, 255), transpose=True)
    plt.imshow(img)
    plt.show()
