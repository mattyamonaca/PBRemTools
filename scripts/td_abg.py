import cv2
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


# Declare Execution Providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Download and host the model
model_path = huggingface_hub.hf_hub_download(
    "skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)

def get_mask(img, s=1024):
    img = (img / 255).astype(np.float32)
    dim = img.shape[2]
    if dim == 4:
        img = img[..., :3]
        dim = 3
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, dim], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw //
              2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

def assign_tile(row, tile_width, tile_height):
    tile_x = row['x_l'] // tile_width
    tile_y = row['y_l'] // tile_height
    return f"tile_{tile_y}_{tile_x}"

def rmbg_fn(img):
    mask = get_mask(img)
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    mask = mask.repeat(3, axis=2)
    return mask, img

def refinement(img, mask, fast, psp_L):
    mask =  cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

    # Fast - Global step only.
    # Smaller L -> Less memory usage; faster in fast mode.
    mask = refiner.refine(img, mask, fast=fast, L=psp_L) 

    return mask


def get_foreground(img, td_abg_enabled, h_split, v_split, n_cluster, alpha, th_rate, cascadePSP_enabled, fast, psp_L):
    if td_abg_enabled == True:
        mask = get_mask(img)
        mask = (mask * 255).astype(np.uint8)
        mask = mask.repeat(3, axis=2)
        if cascadePSP_enabled == True:
            mask =  refinement(img, mask, fast, psp_L)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        df = rgb2df(img)

        image_width = img.shape[1] 
        image_height = img.shape[0] 

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
        img_df.loc[img_df['bg_cls'] == 0, ['a']] = 0
        img_df.loc[img_df['bg_cls'] != 0, ['a']] = 255
        img = df2rgba(img_df)

    if cascadePSP_enabled == True and td_abg_enabled == False:
        mask = get_mask(img)
        mask = (mask * 255).astype(np.uint8)
        refiner = refine.Refiner(device='cuda:0')
        mask = refiner.refine(img, mask, fast=fast, L=psp_L)
        img = np.dstack((img, mask))
    
    if cascadePSP_enabled == False and td_abg_enabled == False:
        mask, img = rmbg_fn(img)

    return mask, img






