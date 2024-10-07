import gradio as gr
from scripts.tools import processing, model_list
from modules import script_callbacks
import modules.shared as shared

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as PBRemTools:
        input_tab_state = gr.State(value=0)
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem(label="Single") as input_tab_single:
                        single_image = gr.Image(type="pil")
                    with gr.TabItem(label="Batch") as input_tab_batch:
                        batch_image = gr.File(label="Batch Images", file_count="multiple", interactive=True, type="file")
                    with gr.TabItem(label="Batch from Dir") as input_tab_dir:
                        input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs)
                        output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs)
                        output_mask_dir = gr.Textbox(label="Output Mask directory", **shared.hide_dirs)
                        show_result = gr.Checkbox(label="Show result images", value=True)
                with gr.Accordion("Mask Setting", open=True):
                    with gr.Tab("Segment Anything & CLIP"):
                        sa_enabled = gr.Checkbox(label="enabled", show_label=True)
                        model_name = gr.Dropdown(label="Model", elem_id="sam_model", choices=model_list,
                                                 value=model_list[0] if len(model_list) > 0 else None)
                        seg_query = gr.Textbox(label = "segmentation prompt", show_label=True)
                        predicted_iou_threshold = gr.Slider(0, 1, value=0.9, step=0.01, label="predicted_iou_threshold", show_label=True)
                        stability_score_threshold = gr.Slider(0, 1, value=0.9, step=0.01, label="stability_score_threshold", show_label=True)
                        clip_threshold =  gr.Slider(0, 1, value=0.1, step=0.01, label="clip_threshold", show_label=True)
                with gr.Accordion("Post Processing", open=True):
                    with gr.Tab("tile division BG Removers"):
                        td_abg_enabled = gr.Checkbox(label="enabled", show_label=True)
                        h_split = gr.Slider(1, 2048, value=256, step=4, label="horizontal split num", show_label=True)
                        v_split = gr.Slider(1, 2048, value=256, step=4, label="vertical split num", show_label=True)
                        
                        n_cluster = gr.Slider(1, 1000, value=500, step=10, label="cluster num", show_label=True)
                        alpha = gr.Slider(1, 255, value=50, step=1, label="alpha threshold", show_label=True)
                        th_rate = gr.Slider(0, 1, value=0.1, step=0.01, label="mask content ratio", show_label=True)
                    with gr.Tab("cascadePSP"):
                        cascadePSP_enabled = gr.Checkbox(label="enabled", show_label=True)
                        fast = gr.Checkbox(label="fast", show_label=True)
                        psp_L = gr.Slider(1, 2048, value=900, step=1, label="Memory usage", show_label=True)

                submit = gr.Button(value="Submit")
            with gr.Row():
                with gr.Column():
                    gallery = gr.Gallery(label="outputs", show_label=True, elem_id="gallery", columns=2, object_fit="contain")

        # 0: single 1: batch 2: batch dir
        input_tab_single.select(fn=lambda: 0, inputs=[], outputs=[input_tab_state])
        input_tab_batch.select(fn=lambda: 1, inputs=[], outputs=[input_tab_state])
        input_tab_dir.select(fn=lambda: 2, inputs=[], outputs=[input_tab_state])
        submit.click(
            processing, 
            inputs=[single_image, batch_image, input_dir, output_dir, output_mask_dir, show_result, input_tab_state, td_abg_enabled, h_split, v_split, n_cluster, alpha, th_rate, cascadePSP_enabled, fast, psp_L, sa_enabled, seg_query, model_name, predicted_iou_threshold, stability_score_threshold, clip_threshold],
            outputs=gallery
        )

    return [(PBRemTools, "PBRemTools", "pbremtools")]
        
script_callbacks.on_ui_tabs(on_ui_tabs)
