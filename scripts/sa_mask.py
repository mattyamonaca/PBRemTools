import os
import urllib
from functools import lru_cache
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import clip
import cv2
import gradio as gr
import numpy as np
import PIL
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

CHECKPOINT_PATH = "./models/sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
MAX_WIDTH = MAX_HEIGHT = 800
CLIP_WIDTH = CLIP_HEIGHT = 300
THRESHOLD = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@lru_cache
def load_mask_generator() -> SamAutomaticMaskGenerator:
    if not os.path.exists(os.path.join(".", CHECKPOINT_PATH)):
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


@lru_cache
def load_clip(
    name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess


def adjust_image_size(image: np.ndarray) -> np.ndarray:
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


def filter_masks(
    image: np.ndarray,
    masks: List[Dict[str, Any]],
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    query: str,
    clip_threshold: float,
) -> List[Dict[str, Any]]:
    cropped_masks: List[PIL.Image.Image] = []
    filtered_masks: List[Dict[str, Any]] = []

    for mask in masks:
        if (
            mask["predicted_iou"] < predicted_iou_threshold
            or mask["stability_score"] < stability_score_threshold
        ):
            continue

        filtered_masks.append(mask)

        x, y, w, h = mask["bbox"]
        crop = image[y : y + h, x : x + w]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = PIL.Image.fromarray(np.uint8(crop * 255)).convert("RGB")
        crop.resize((CLIP_WIDTH, CLIP_HEIGHT))
        cropped_masks.append(crop)

    if query and filtered_masks:
        scores = get_scores(cropped_masks, query)
        filtered_masks = [
            filtered_masks[i]
            for i, score in enumerate(scores)
            if score > clip_threshold
        ]

    return filtered_masks


def draw_masks(image, masks, alpha: float = 0.7) -> np.ndarray:
    for mask in masks:
        color = [randint(127, 255) for _ in range(3)]

        # draw mask overlay
        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(
            np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    return image


def segment(predicted_iou_threshold, stability_score_threshold, clip_threshold, image, query):
    mask_generator = load_mask_generator()
    image = adjust_image_size(image)
    masks = mask_generator.generate(image)
    masks = filter_masks(
        image,
        masks,
        predicted_iou_threshold,
        stability_score_threshold,
        query,
        clip_threshold,
    )
    image = draw_masks(image, masks)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(np.uint8(image)).convert("RGB")
    return masks


def get_sa_mask(image, query):
    predicted_iou_threshold = 0.9
    stability_score_threshold = 0.9
    clip_threshold = 0.1
    masks = segment(predicted_iou_threshold, stability_score_threshold, clip_threshold, image, query)
    mask_list = []
    for mask in masks:
        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        colored_mask = np.where(colored_mask, 0, 255)
        mask_list.append(colored_mask)
    combined_mask = np.minimum.reduce(mask_list)

    return 255 - combined_mask
    