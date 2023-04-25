import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from scripts.convertor import rgb2df, df2rgba

import gradio as gr
import huggingface_hub
import onnxruntime as rt
import copy
from PIL import Image

import segmentation_refinement as refine

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
from nptyping import NDArray, Shape, UInt8


# Declare Execution Providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Download and host the model
model_path = huggingface_hub.hf_hub_download(
    "skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)

def segment_anime(image: NDArray[Shape["Height,Width,[r,g,b]"], UInt8], s:int=1024) -> List[NDArray[Shape["Height,Width"], UInt8]]:
    img = (image / 255.).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw //
              2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))

    # Convert back to uint8
    mask = np.clip(mask, 0, 1) * 255
    mask = np.rint(mask).astype(np.uint8)
    return [mask]

def assign_tile(row: pd.DataFrame, tile_width: int, tile_height:int) -> str:
    tile_x = row['x_l'] // tile_width
    tile_y = row['y_l'] // tile_height
    return f"tile_{tile_y}_{tile_x}"

def cascadePSP(image: NDArray[Shape["Height,Width,[r,g,b]"], UInt8],
               masks: List[NDArray[Shape["Height,Width"], UInt8]],
               extended_res: bool, resolution: int) -> List[NDArray[Shape["Height,Width"], UInt8]]:
    refiner = refine.Refiner(device='cuda:0') # device can also be cpu'
    masks_out = []
    for mask in masks:
        # Fast - Global step only.
        # Smaller L -> Less memory usage; faster in fast mode.
        mask_out = refiner.refine(image, mask, fast=not extended_res, L=resolution)

        masks_out.append(mask_out)
    with torch.no_grad():
        torch.cuda.empty_cache()
    return masks_out

def td_abg(image: NDArray[Shape["Height,Width,[r,g,b]"], UInt8],
           masks: List[NDArray[Shape["Height,Width"], UInt8]],
           h_split: int, v_split: int, n_cluster: int, alpha: float, th_rate: float) -> List[NDArray[Shape["Height,Width"], UInt8]]:
    masks_out = []
    image_width = image.shape[1] 
    image_height = image.shape[0] 
    for mask in masks:
        df = rgb2df(image)
        num_horizontal_splits = h_split
        num_vertical_splits = v_split
        tile_width = image_width // num_horizontal_splits
        tile_height = image_height // num_vertical_splits

        df['tile'] = df.apply(assign_tile, args=(tile_width, tile_height), axis=1)

        cls = MiniBatchKMeans(n_clusters=n_cluster, batch_size=100)
        cls.fit(df[["r","g","b"]])
        df["label"] = cls.labels_

        mask_df = rgb2df(mask)
        mask_df['bg_label'] = (mask_df['r'] > alpha) & (mask_df['g'] > alpha) & (mask_df['b'] > alpha)
        

        img_df = df.copy()
        img_df["bg_label"] = mask_df["bg_label"]
        img_df["label"] = img_df["label"].astype(str) + "-" + img_df["tile"]
        bg_rate = img_df.groupby("label").sum()["bg_label"]/img_df.groupby("label").count()["bg_label"]
        img_df['bg_cls'] = (img_df['label'].isin(bg_rate[bg_rate > th_rate].index)).astype(int)
        #img_df.loc[img_df['bg_cls'] == 0, ['a']] = 0
        #img_df.loc[img_df['bg_cls'] != 0, ['a']] = 255
        #img = df2rgba(img_df)

        mask_df.loc[img_df['bg_cls'] == 0, ['r', 'g', 'b', 'a']] = (255, 255, 255, 0)
        mask_df.loc[img_df['bg_cls'] != 0, ['r', 'g', 'b', 'a']] = (0, 0, 0, 0)
        mask = df2rgba(mask_df)
        
        masks_out.append(mask)
    
    return masks_out

