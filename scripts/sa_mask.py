import os
import urllib
from functools import lru_cache
from random import randint
from typing import Any, Callable, Dict, List, Tuple, TypeVar

import clip
import cv2
import gradio as gr
import numpy as np
import PIL
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.modeling import Sam
from collections import OrderedDict
from nptyping import NDArray, Shape, UInt8, Float32

try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir

from modules.safe import unsafe_torch_load, load
from modules.devices import device

from scripts.convertor import cv2pil

model_cache: OrderedDict = OrderedDict()
sam_model_dir = os.path.join(
    extensions_dir, "PBRemTools/models/")
model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']

MAX_WIDTH = MAX_HEIGHT = 800
CLIP_WIDTH = CLIP_HEIGHT = 300
THRESHOLD = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sam_model(sam_checkpoint: str) -> Sam:
    model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
    sam_checkpoint = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    torch.load = load
    return sam


def load_mask_generator(model_name: str) -> SamAutomaticMaskGenerator:
    sam = load_sam_model(model_name)
    mask_generator = SamAutomaticMaskGenerator(sam)
    torch.load = load
    return mask_generator


def load_clip(
    name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess

# Any->Any isn't correct, but see https://github.com/ramonhagenaars/nptyping/issues/107
def adjust_image_size(image: NDArray[Shape["Height,Width,..."], Any]) -> NDArray[Shape["*,*,..."], Any]:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image


@torch.no_grad()
def get_scores(crops: List[PIL.Image.Image], query: str) -> torch.Tensor:
    model, preprocess = load_clip()
    preprocessed = [preprocess(crop) for crop in crops]
    preprocessed = torch.stack(preprocessed).to(device)
    token = clip.tokenize(query).to(device)
    img_features = model.encode_image(preprocessed)
    txt_features = model.encode_text(token)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    txt_features /= txt_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * img_features @ txt_features.T
    return probs[:, 0].softmax(dim=0)


def bbox(image: NDArray[Shape["Height,Width,..."], Any]) -> Tuple[int, int, int, int]:
    rows = np.any(image, axis=0)
    cols = np.any(image, axis=1)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return xmin, xmax, ymin, ymax


def union_masks(image: NDArray[Shape["Height,Width, [r,g,b]"], UInt8], masks: List[NDArray[Shape["Height,Width"], UInt8]]) -> NDArray[Shape["Height,Width,[r,g,b]"], UInt8]:
    union_mask = np.zeros(image.shape[:-1], dtype=np.uint8)
    for mask in masks:
        union_mask = np.maximum(union_mask, mask)
    return union_mask


def do_clip(
    image: NDArray[Shape["Height,Width,[r,g,b]"], UInt8],
    masks: List[NDArray[Shape["Height,Width"], UInt8]],
    seg_query: str,
    clip_threshold: float,
) -> List[NDArray[Shape["Height,Width"], UInt8]]:
    cropped_masks: List[PIL.Image.Image] = []

    for mask in masks:
        xmin, xmax, ymin, ymax = bbox(mask)
        print("do_clip", image, image.shape, image.dtype, mask, mask.shape, mask.dtype, xmin, xmax, ymin, ymax)
        crop = image[ymin:ymax, xmin:xmax]
        crop = PIL.Image.fromarray(crop, mode="RGB")
        crop.resize((CLIP_WIDTH, CLIP_HEIGHT))
        cropped_masks.append(crop)

    scores = get_scores(cropped_masks, seg_query)
    print(scores)
    masks = [
        masks[i]
        for i, score in enumerate(scores)
        if score >= clip_threshold
    ]

    masks = [union_masks(image, masks)]

    return masks


def masked_images_to_pil(
    image: NDArray[Shape["Height,Width,[r,g,b]"], UInt8],
    masks: List[NDArray[Shape["Height,Width"], UInt8]],
) -> List[PIL.Image.Image]:
    ret = []
    for mask in masks:
        print("masked_images_to_pil", image, image.shape, image.dtype, mask, mask.shape, mask.dtype)
        s = list(image.shape)
        s[-1] = 4
        nimage = np.empty(s, dtype=image.dtype)
        nimage[:,:,:3] = image
        nimage[:,:,3] = mask

        ret.append(PIL.Image.fromarray(nimage, mode="RGBA"))
    return ret


def masks_to_pil(
        masks: List[NDArray[Shape["Height,Width"], UInt8]]) -> List[PIL.Image.Image]:
    ret = []
    for mask in masks:
        print("masks_to_pil", mask, mask.shape, mask.dtype)
        ret.append(PIL.Image.fromarray(mask, mode="L").convert("RGB"))
    return ret


#def draw_masks(image: np.ndarray, masks: Tuple[np.ndarray], alpha: float = 0.7) -> np.ndarray:
#    for mask in masks:
#        color = [randint(127, 255) for _ in range(3)]
#
#        # draw mask overlay
#        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
#        colored_mask = np.moveaxis(colored_mask, 0, -1)
#        masked: np.ma.MaskedArray = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
#        image_overlay = masked.filled()
#        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
#
#        # draw contour
#        contours, _ = cv2.findContours(
#            np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#        )
#        cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
#    return image


def segment_anything_now(image: NDArray[Shape["Height,Width, [r,g,b]"], UInt8],
                         model_name: str, min_predicted_iou: float, min_stability_score: float) -> List[NDArray[Shape["Height,Width"], UInt8]]:
    mask_generator = load_mask_generator(model_name)
    image = adjust_image_size(image)
    masks = mask_generator.generate(image)
    mask_list = []
    for mask in masks:
        if (
            mask["predicted_iou"] < min_predicted_iou
            or mask["stability_score"] < min_stability_score
        ):
            continue
        colored_mask = mask["segmentation"].astype(np.uint8) * 255
        print("segment_anything_now", colored_mask, colored_mask.shape, colored_mask.dtype)
        mask_list.append(colored_mask)
    return mask_list

