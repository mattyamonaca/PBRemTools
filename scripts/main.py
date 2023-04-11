import os
import io
import json
import numpy as np
import cv2

import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks

from scripts.td_abg import get_foreground
from scripts.convertor import pil2cv
from modules.paths_internal import extensions_dir
from collections import OrderedDict


model_cache = OrderedDict()
sam_model_dir = os.path.join(
    extensions_dir, "PBRemTools/models/")
model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']


def processing(input_image, td_abg_enabled, h_split, v_split, n_cluster, alpha, th_rate, cascadePSP_enabled, fast, psp_L, sa_enabled, seg_query, model_name, predicted_iou_threshold, stability_score_threshold, clip_threshold):
    image = pil2cv(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask, image = get_foreground(image, td_abg_enabled, h_split, v_split, n_cluster, alpha, th_rate, cascadePSP_enabled, fast, psp_L, sa_enabled, seg_query, model_name, predicted_iou_threshold, stability_score_threshold, clip_threshold)
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

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as PBRemTools:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil")
                with gr.Accordion("Mask Setting", open=True):
                  with gr.Accordion("Segment Anithing & CLIP", open=True):  
                    with gr.Accordion("Segment Anithing & CLIP", open=True):
                      sa_enabled = gr.Checkbox(label="enabled", show_label=True)
                      model_name = gr.Dropdown(label="Model", elem_id="sam_model", choices=model_list,
                                          value=model_list[0] if len(model_list) > 0 else None)
                      seg_query = gr.Textbox(label = "segmentation prompt", show_label=True)
                      predicted_iou_threshold = gr.Slider(0, 1, value=0.9, step=0.01, label="predicted_iou_threshold", show_label=True)
                      stability_score_threshold = gr.Slider(0, 1, value=0.9, step=0.01, label="stability_score_threshold", show_label=True)
                      clip_threshold =  gr.Slider(0, 1, value=0.1, step=0.01, label="clip_threshold", show_label=True)

                with gr.Accordion("Post Processing", open=True):
                  with gr.Accordion("tile division BG Removers", open=True):
                      td_abg_enabled = gr.Checkbox(label="enabled", show_label=True)
                      h_split = gr.Slider(1, 2048, value=256, step=4, label="horizontal split num", show_label=True)
                      v_split = gr.Slider(1, 2048, value=256, step=4, label="vertical split num", show_label=True)
                      
                      n_cluster = gr.Slider(1, 1000, value=500, step=10, label="cluster num", show_label=True)
                      alpha = gr.Slider(1, 255, value=50, step=1, label="alpha threshold", show_label=True)
                      th_rate = gr.Slider(0, 1, value=0.1, step=0.01, label="mask content ratio", show_label=True)
                        
                  with gr.Accordion("cascadePSP", open=True):
                      cascadePSP_enabled = gr.Checkbox(label="enabled", show_label=True)
                      fast = gr.Checkbox(label="fast", show_label=True)
                      psp_L = gr.Slider(1, 2048, value=900, step=1, label="Memory usage", show_label=True)

                submit = gr.Button(value="Submit")
            with gr.Row():
                with gr.Column():
                    with gr.Tab("output"):
                        output_img = gr.Image()
                    with gr.Tab("mask"):
                        output_mask = gr.Image()
        submit.click(
            processing, 
            inputs=[input_image, td_abg_enabled, h_split, v_split, n_cluster, alpha, th_rate, cascadePSP_enabled, fast, psp_L, sa_enabled, seg_query, model_name, predicted_iou_threshold, stability_score_threshold, clip_threshold], 
            outputs=[output_img, output_mask]
        )

    return [(PBRemTools, "PBRemTools", "pbremtools")]

script_callbacks.on_ui_tabs(on_ui_tabs)