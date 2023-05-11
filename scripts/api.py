from typing import List

import gradio as gr
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel, Field

from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from scripts.main import process_image, model_list


class PBRRemRequest(BaseModel):
    img: str = Field(..., description='Image to process, base64 encoded')

    td_abg_enabled: bool = False
    h_split: int = 256
    v_split: int = 256
    n_cluster: int = 500
    alpha: int = 50
    th_rate: float = 0.1

    cascadePSP_enabled: bool = False
    fast: bool = False
    psp_L: int = 900

    sa_enabled: bool = False
    model_name: str = Field('sam_vit_h_4b8939.pth',
                            description='SAM model name')
    query: str = Field('', description='segmentation prompt')
    predicted_iou_threshold: float = 0.9
    stability_score_threshold: float = 0.9
    clip_threshold: float = 0.1


class PBRRemResponse(BaseModel):
    img: str
    mask: str


def pbrem_api(_: gr.Blocks, app: FastAPI):

    @app.get("/pbrem/sam-model", description='query available SAM model', response_model=List[str])
    async def get_pbrem_sam_model() -> List[str]:
        return model_list

    @app.post('/pbrem/predict', description='process image', response_model=PBRRemResponse)
    async def post_pbrem_predict(payload: PBRRemRequest) -> PBRRemResponse:
        print(f"PBRemTools API /pbrem/predict received request")
        input_image = decode_base64_to_image(payload.img)
        output_image, mask = process_image(
            input_image,
            payload.td_abg_enabled, payload.h_split, payload.v_split, payload.n_cluster, payload.alpha, payload.th_rate,
            payload.cascadePSP_enabled, payload.fast, payload.psp_L,
            payload.sa_enabled, payload.query, payload.model_name, payload.predicted_iou_threshold, payload.stability_score_threshold, payload.clip_threshold
        )
        base64_img = encode_pil_to_base64(Image.fromarray(output_image))
        base64_mask = encode_pil_to_base64(Image.fromarray(mask))
        # Compatible: low version webui will return base64 string directly
        if not isinstance(base64_img, str):
            base64_img = base64_img.decode()
        if not isinstance(base64_mask, str):
            base64_mask = base64_mask.decode()
        return PBRRemResponse(img=base64_img, mask=base64_mask)


try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(pbrem_api)
except Exception as e:
    print(e)
    print("PBRemTools API failed to initialize")
