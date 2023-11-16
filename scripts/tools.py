import os
import io
import json
import numpy as np
import cv2
import re
import modules.scripts as scripts
import modules.shared as shared
from modules import images

from scripts.td_abg import get_foreground
from scripts.convertor import pil2cv
from scripts.batch_dir import save_image_dir, modify_basename

try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir

from collections import OrderedDict
from PIL import Image

model_cache = OrderedDict()
models_path = shared.models_path
sams_dir = os.path.join(models_path, "sam")
if os.path.exists(sams_dir):
    sam_model_dir = sams_dir
else:
    sam_model_dir = os.path.join(
        extensions_dir, "PBRemTools/models/")
model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']

def processing(single_image, batch_image, input_dir, output_dir, output_mask_dir, show_result, input_tab_state, *rem_args):
    # 0: single
    if (input_tab_state == 0):
        processed = process_image(single_image, *rem_args)
        return processed
    # 1: batch
    elif (input_tab_state == 1):
        processed = []
        for i in batch_image:
            image = Image.open(i)
            base, mask = process_image(image, *rem_args)
            processed.append(base)
            processed.append(mask)
        return processed
    # 2: batch dir (or other)
    else:
        processed = []
        files = shared.listfiles(input_dir)
        for f in files:
            try:
                image = Image.open(f)
            except Exception:
                continue
            base, mask = process_image(image, *rem_args)
            processed.append(base)
            processed.append(mask)
            if output_dir != "":
                basename = os.path.splitext(os.path.basename(f))[0]
                ext = os.path.splitext(f)[1][1:]
                save_image_dir(
                    Image.fromarray(base),
                    path=output_dir,
                    basename=basename,
                    extension="png",
                )
            if output_mask_dir != "":
                basename = modify_basename(basename)
                save_image_dir(
                    Image.fromarray(mask),
                    path=output_mask_dir,
                    basename=basename,
                    extension="png",
                )
        if (show_result):
            return processed
        else:
            return None

def process_image(target_image, *rem_args):
    image = pil2cv(target_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask, image = get_foreground(image, *rem_args)
    return image, mask

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "PBRemTools"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

