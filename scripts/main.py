import os
import io
import json
import numpy as np
import cv2

import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks

from scripts.sa_mask import segment_anything_now, do_clip, masked_images_to_pil, masks_to_pil
from scripts.td_abg import segment_anime, td_abg, cascadePSP
from scripts.convertor import pil2cv
try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir

from collections import namedtuple, OrderedDict
import logging

from typing import Dict, Any, NamedTuple, Union, Set, TypeVar, Tuple, List, Sequence
try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard
from nptyping import NDArray, Shape, UInt8

import PIL

logger = logging.getLogger(__name__)


model_cache: OrderedDict = OrderedDict()
sam_model_dir = os.path.join(
    extensions_dir, "PBRemTools/models/")
model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']

SANSettings = NamedTuple("SANSettings", (
                         ("model_name", Union[gr.Image, PIL.Image.Image]),
                         ("predicted_iou_threshold", Union[gr.Slider, float]),
                         ("stability_score_threshold", Union[gr.Slider, float])))
AnimeSegSettings = NamedTuple("AnimeSegSettings", ())

SegmentationSettings = NamedTuple("SegmentationSettings", (
                                 ("tab", Union[gr.State, int]),
                                 ("san", SANSettings),
                                 ("anime_seg", AnimeSegSettings)))

CLIPSettings = NamedTuple("CLIPSettings", (
                          ("enabled", Union[gr.Checkbox, bool]),
                          ("seg_query", Union[gr.Textbox, str]),
                          ("clip_threshold", Union[gr.Slider, float])))

TdAbgSettings = NamedTuple("TdAbgSettings", (
                          ("h_split", Union[gr.Slider, int]),
                          ("v_split", Union[gr.Slider, int]),
                          ("n_cluster", Union[gr.Slider, int]),
                          ("alpha", Union[gr.Slider, float]),
                          ("th_rate", Union[gr.Slider, float])))

CascadePSPSettings = NamedTuple("CascadePSPSettings", (
                               ("extended_res", Union[gr.Checkbox, bool]),
                               ("resolution", Union[gr.Slider, int])))
PostprocessorSettings = NamedTuple("PostprocessorSettings", (
                                  ("accordion", Union[gr.Accordion, bool]),
                                  ("tab", Union[gr.State, int]),
                                  ("move_up", Union[gr.Button, bool]),
                                  ("move_down", Union[gr.Button, bool]),
                                  ("clone_above", Union[gr.Button, bool]),
                                  ("delete", Union[gr.Button, bool]),
                                  ("td_abg", TdAbgSettings),
                                  ("cascadePSP", CascadePSPSettings)))

PostprocessingSettings = NamedTuple("PostprocessingSettings", (
                                   ("append", Union[gr.Button, bool]),
                                   ("num_enabled", Union[gr.State, bool]),
                                   ("postprocessors", Tuple[PostprocessorSettings])))

TopSettings = NamedTuple("TopSettings", (
                        ("input_image", Union[gr.Image, bool]),
                        ("segmentation", SegmentationSettings),
                        ("clip", CLIPSettings),
                        ("postprocessing", PostprocessingSettings),
                        ("submit", Union[gr.Button, bool])))

ImageData = namedtuple("ImageData", ("image", "masks"))

FlatArgs = Dict[gr.blocks.Block, Any]
T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
L = TypeVar("L")
V = TypeVar("V")
W = TypeVar("W")


def processing(flatargs: FlatArgs, *, top: TopSettings) -> List[PIL.Image.Image]:
    vals = unflatten_args(flatargs=flatargs, top=top)
    print(f"processing(flatargs={flatargs}, top={top}, vals={vals})")

    if vals is None:
        raise ValueError(f"Invalid vals flatargs={flatargs} top={top}")

    image = pil2cv(vals.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    seg = vals.segmentation
    print(f"seg.tab={seg.tab}")
    if seg.tab == "san":
        san = seg.san
        
        masks = segment_anything_now(image=image,
            model_name=san.model_name,
            min_predicted_iou=san.predicted_iou_threshold,
            min_stability_score=san.stability_score_threshold)
        image = ImageData(image=image, masks=tuple(masks))
    elif seg.tab == "anime_seg":
        anime_seg = seg.anime_seg
        masks = segment_anime(image=image)
        image = ImageData(image=image, masks=tuple(masks))
    #elif seg == "upload_mask":
    #   # TODO: upload mask
    else:
        raise ValueError(f"Invalid segmentation tab {seg.tab}")

    if vals.clip.enabled:
        clip = vals.clip
        masks = do_clip(image=image.image,
                        masks=masks,
                        seg_query=clip.seg_query,
                        clip_threshold=clip.clip_threshold)
        image = image._replace(masks=masks)

    for p in vals.postprocessing.postprocessors[:vals.postprocessing.num_enabled]:
        if p.tab == "td_abg":
            t = p.td_abg
            image = image._replace(masks=tuple(td_abg(image=image.image,
                                                      masks=image.masks,
                                                      h_split=t.h_split,
                                                      v_split=t.v_split,
                                                      n_cluster=t.n_cluster,
                                                      alpha=t.alpha,
                                                      th_rate=t.th_rate)))
        elif p.tab == "cascadePSP":
            c = p.cascadePSP
            image = image._replace(masks=tuple(cascadePSP(image=image.image,
                                                          masks=image.masks,
                                                          extended_res=c.extended_res,
                                                          resolution=c.resolution)))
        else:
            raise ValueError(f"Invalid postprocessing tab {p.tab}")

    return masked_images_to_pil(image.image, image.masks) + masks_to_pil(image.masks)


class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "PBRemTools"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()



"""
    d[k] = v, but with assertion that key is not in dict
"""
def add_new(d, k, v):
    assert k not in d, f"Duplicate key {k} - was {d[k]} adding {v}"
    d[k] = v
    return d

"""
    return {**a, **b}, but with assertion that keys do not overlap
"""
def merged_dicts(a: Dict[K, V], b: Dict[L, W]) -> Dict[Union[K, L], Union[V, W]]:
    t: Dict[Union[K, L], Union[V, W]] = {k: v for k, v in a.items()}
    for k,v in b.items():
        add_new(t, k, v)
    return t
"""
    return a | b, but with assertion that keys do not overlap
"""
def merged_sets(a: Set[T], b: Set[U]) -> Set[Union[T, U]]:
    ret = a | b
    assert len(ret) == len(a) + len(b), f"Duplicate key(s) {a.union(b)}"
    return ret


def is_any_named_tuple(o: Any) -> TypeGuard[NamedTuple]:
    return isinstance(o, tuple) and hasattr(o, "_asdict") and hasattr(o, "_fields")


"""
    Turns a potentially-nested namedtuple of components into a flat set.
"""
def get_components(c: Any, sofar: Union[Set[gr.components.Component], None]=None) -> Set[gr.components.Component]:
    if sofar is None:
        sofar = set()
    if is_any_named_tuple(c):
        for v in c._asdict().values():
            get_components(v, sofar)
    elif isinstance(c, list) or isinstance(c, tuple):
        for v in c:
            get_components(v, sofar)
    elif isinstance(c, gr.components.Component):
        assert c not in sofar, f"Duplicate component {c}"
        sofar.add(c)
    elif isinstance(c, gr.blocks.Block):
        pass
    else:
        logger.warn(f"Unknown component {c} of type {type(c)}")
    return sofar


"""
    top: a potentially-nested namedtuple of components
    flatargs: a dictionary from components to values

    output: a potentially-nested namedtuple of values
"""
def unflatten_args(flatargs: FlatArgs, top: T) -> Union[T, None]:
    if is_any_named_tuple(top):
        return top._replace(**{k: unflatten_args(flatargs, v) for k, v in top._asdict().items()}) # type: ignore
    elif isinstance(top, list) or isinstance(top, tuple):
        nxt = [unflatten_args(flatargs, v) for v in top]
        if isinstance(top, tuple):
            return tuple(nxt) # type: ignore
        return nxt # type: ignore
    elif isinstance(top, gr.components.Component):
        return flatargs[top]
    elif isinstance(top, gr.blocks.Block):
        return None
    else:
        logger.warn(f"Unknown component {top} of type {type(top)}")
        return None

"""
    Moves the values in 'a' to 'b' instead
"""
def nested_move(a: Any, b: Any, args: FlatArgs, sofar:Union[Dict[gr.blocks.Block, Any], None]=None) -> Dict[gr.blocks.Block, Any]:
    if sofar is None:
        sofar = {}
    assert type(a) == type(b), f"Cannot swap {a} and {b} ({type(a)} != {type(b)})"
    if is_any_named_tuple(a):
        for aa, bb in zip(a._asdict().items(), b._asdict().items()):
            ak, av = aa
            bk, bv = bb
            assert ak == bk, f"Key mismatch: {aa} != {bb}"
            nested_move(av, bv, args, sofar)
    elif isinstance(a, list) or isinstance(a, tuple):
        for av, bv in zip(a, b):
            nested_move(av, bv, args, sofar)
    elif isinstance(a, gr.components.Component):
        add_new(sofar, b, args[a])
    elif isinstance(a, gr.blocks.Block):
        pass
    else:
        logger.warn(f"Unknown component {a} of type {type(a)}")
    return sofar


def update_buttons(ret: Dict[gr.blocks.Block, Any], postprocessing: PostprocessingSettings, num_enabled: int) -> Dict[gr.blocks.Block, Any]:
    print(f"update_buttons(ret={ret}, postprocessing={postprocessing})")
    # TODO: more efficient update
    for i, p in enumerate(postprocessing.postprocessors):
        ret[p.move_up] = gr.update(interactive=(i > 0))
        ret[p.move_down] = gr.update(interactive=(i < num_enabled-1))
        ret[p.clone_above] = gr.update(interactive=num_enabled < len(postprocessing.postprocessors))
    add_new(ret, postprocessing.append, gr.update(interactive=num_enabled < len(postprocessing.postprocessors)))
    print(f"update_buttons(ret={ret}, postprocessing={postprocessing}) -> {ret}")
    return ret


def swap_postprocessors(a: Any, b: Any, args: FlatArgs, postprocessing: PostprocessingSettings) -> Dict[gr.blocks.Block, Any]:
    print(f"swap_postprocessors(a={a}, b={b}, args={args}, postprocessing={postprocessing})")
    num_enabled = args[postprocessing.num_enabled]
    if a >= num_enabled:
        logger.warn(f"Cannot swap postprocessor {a} past end {num_enabled}")
        return args
    if b >= num_enabled:
        logger.warn(f"Cannot swap postprocessor {b} past end {num_enabled}")
        return args
    pa, pb = postprocessing.postprocessors[a], postprocessing.postprocessors[b]
    qa, qb = nested_move(pa, pb, args), nested_move(pb, pa, args)
    ret = merged_dicts(qa, qb)

    update_buttons(ret, postprocessing, num_enabled)

    print(f"swap_postprocessors(a={a}, b={b}, args={args}, postprocessing={postprocessing}) -> {ret}")

    return ret

def insert_postprocessor(index: int, args: FlatArgs, postprocessing: PostprocessingSettings) -> Dict[gr.blocks.Block, Any]:
    num_enabled = postprocessing.num_enabled
    if args[num_enabled] >= len(postprocessing.postprocessors):
        logger.warn("Cannot insert postprocessor; all slots full")
        return args
    
    ret: Dict[gr.blocks.Block, Any] = {}
    for i in range(index+1, len(postprocessing.postprocessors)):
        nested_move(postprocessing.postprocessors[i-1], postprocessing.postprocessors[i], args, ret)

    # TODO: reset values to defaults
    # You'd think there'd be a method for this...

    add_new(ret, num_enabled, args[num_enabled] + 1)

    for i, p in enumerate(postprocessing.postprocessors):
        add_new(ret, p.accordion, gr.update(visible=(i < ret[num_enabled])))

    update_buttons(ret, postprocessing, ret[num_enabled])

    return ret

def append_postprocessor(args: FlatArgs, postprocessing: PostprocessingSettings) -> Dict[gr.blocks.Block, Any]:
    num_enabled = args[postprocessing.num_enabled]
    return insert_postprocessor(num_enabled, args, postprocessing)


def delete_postprocessor(index: int, args: FlatArgs, postprocessing: PostprocessingSettings) -> Dict[gr.blocks.Block, Any]:
    num_enabled = postprocessing.num_enabled
    postprocessors = postprocessing.postprocessors
    if args[num_enabled] <= 0:
        logger.warn("Cannot delete postprocessor; all slots full")
        return args
    
    ret: Dict[gr.blocks.Block, Any] = {}
    for i in range(index+1, len(postprocessors)):
        nested_move(postprocessing.postprocessors[i], postprocessing.postprocessors[i-1], args, ret)
    
    # TODO: reset values to defaults
    # You'd think there'd be a method for this...
    
    add_new(ret, num_enabled, args[num_enabled] - 1)

    for i, p in enumerate(postprocessors):
        add_new(ret, p.accordion, gr.update(visible=(i < ret[num_enabled])))

    update_buttons(ret, postprocessing, ret[num_enabled])

    return ret


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as PBRemTools:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil")

                with gr.Accordion("Segmentation", open=True):
                    cur_tab = gr.State("san")
                    with gr.Tabs(selected="san") as tabs:
                        with gr.Tab("Segment Anything Now", id="san") as tab:
                            tab.select(lambda: "san", inputs=None, outputs=[cur_tab])
                            gr.Checkbox(label="Segment Anything Now", show_label=True, interactive=False, value=True)
                            model_name = gr.Dropdown(label="Model", elem_id="sam_model", choices=model_list,
                                                value=model_list[0] if len(model_list) > 0 else None)
                            predicted_iou_threshold = gr.Slider(0, 1, value=0.9, step=0.01, label="Min Predicted IOU", show_label=True)
                            stability_score_threshold = gr.Slider(0, 1, value=0.9, step=0.01, label="Min Stability Score", show_label=True)
                            san = SANSettings(model_name=model_name,
                                              predicted_iou_threshold=predicted_iou_threshold,
                                              stability_score_threshold=stability_score_threshold)
                        with gr.Tab("Anime Segmentation", id="anime_seg") as tab:
                            tab.select(lambda: "anime_seg", inputs=None, outputs=[cur_tab])
                            gr.Checkbox(label="Anime Segmentation", show_label=True, interactive=False, value=True)
                            anime_seg = AnimeSegSettings()
                        #with gr.Tab("Upload Mask") as tab:
                        #    san.select(lambda: "upload_mask", inputs=None, outputs=cur_tab)
                        #    upload_mask = UploadMaskSettings()
                    
                    segmentation = SegmentationSettings(tab=cur_tab, san=san, anime_seg=anime_seg#, upload_mask=upload_mask
                        )

                
                with gr.Accordion("CLIP", open=True):
                    enabled = gr.Checkbox(label="Enabled", show_label=True)
                    seg_query = gr.Textbox(label = "Prompt", show_label=True)
                    clip_threshold =  gr.Slider(0, 1, value=0.1, step=0.01, label="Clip Threshold", show_label=True)
                    clip = CLIPSettings(enabled=enabled,
                        seg_query=seg_query,
                        clip_threshold=clip_threshold)

                with gr.Accordion("Postprocessing", open=True):
                    MAX_POSTPROCESSORS = 2

                    postprocessors = []
                    accordions = set()
                    for i in range(MAX_POSTPROCESSORS):
                        with gr.Accordion(f"Postprocessor {i}", open=True, visible=False) as accordion:
                            accordions.add(accordion)
                            with gr.Row():
                                move_up = gr.Button(value="Move Up", interactive=(i > 0))
                                move_down = gr.Button(value="Move Down", interactive=False)
                                clone_above = gr.Button(value="Clone Above")
                                delete = gr.Button(value="Delete")

                            cur_tab = gr.State("cascadePSP")
                            with gr.Tabs(selected="cascadePSP") as tabs:
                                with gr.Tab("Tile Division ABG", id="td_abg") as tab:
                                    tab.select(lambda: "td_abg", inputs=None, outputs=[cur_tab])
                                    h_split = gr.Slider(1, 2048, value=256, step=4, label="# of Horizontal Splits", show_label=True)
                                    v_split = gr.Slider(1, 2048, value=256, step=4, label="# of Vertical Splits", show_label=True)
                                    
                                    n_cluster = gr.Slider(1, 1000, value=500, step=10, label="# of Clusters", show_label=True)
                                    alpha = gr.Slider(1, 255, value=100, step=1, label="Alpha Threshold", show_label=True)
                                    th_rate = gr.Slider(0, 1, value=0.1, step=0.01, label="Mask Content Ratio", show_label=True)

                                    td_abg = TdAbgSettings(h_split=h_split,
                                                           v_split=v_split,
                                                           n_cluster=n_cluster,
                                                           alpha=alpha,
                                                           th_rate=th_rate)

                                with gr.Tab("cascadePSP", id="cascadePSP") as tab:
                                    tab.select(lambda: "cascadePSP", inputs=None, outputs=[cur_tab])
                                    extended_res = gr.Checkbox(label="Extended Resolution", show_label=True)
                                    resolution = gr.Slider(1, 2048, value=900, step=1, label="Resolution", show_label=True)

                                    cascadePSP = CascadePSPSettings(extended_res=extended_res, resolution=resolution)

                            postprocessors.append(PostprocessorSettings(tab=cur_tab,
                                                                        accordion=accordion,
                                                                        move_up=move_up,
                                                                        move_down=move_down,
                                                                        clone_above=clone_above,
                                                                        delete=delete,
                                                                        td_abg=td_abg,
                                                                        cascadePSP=cascadePSP))

                    num_enabled = gr.State(0)
                    append = gr.Button(value="Append New Postprocessor")

                    postprocessors = tuple(postprocessors) # freeze list into tuple now that we're done constructing


                    postprocessing = PostprocessingSettings(append=append, num_enabled=num_enabled, postprocessors=postprocessors)
                    # The below is all very inefficient, but meh.
                    args = get_components(postprocessing)
                    oargs = merged_sets(args, accordions)
                    for i in range(1, MAX_POSTPROCESSORS):
                        postprocessors[i].move_up.click(
                            lambda args, i=i, postprocessing=postprocessing: swap_postprocessors(i-1, i, args, postprocessing),
                            inputs=args,
                            outputs=args
                        )
                        postprocessors[i-1].move_down.click(
                            lambda args, i=i, postprocessing=postprocessing: swap_postprocessors(i-1, i, args, postprocessing),
                            inputs=args,
                            outputs=args
                        )
                    for i in range(MAX_POSTPROCESSORS):
                        postprocessors[i].delete.click(
                            fn = (lambda args, i=i, postprocessing=postprocessing: delete_postprocessor(i, args, postprocessing)),
                            inputs = args,
                            outputs = oargs
                        )
                        postprocessors[i].clone_above.click(
                            fn = (lambda args, i=i, postprocessing=postprocessing: insert_postprocessor(i, args, postprocessing)),
                            inputs = args,
                            outputs = oargs
                        )
                    append.click(
                        fn = (lambda args, postprocessing=postprocessing: append_postprocessor(args, postprocessing)),
                        inputs = args,
                        outputs = oargs
                    )
                submit = gr.Button(value="Submit")
            with gr.Row():
                with gr.Column():
                    gallery = gr.Gallery(label="outputs", show_label=True, elem_id="gallery").style(grid=2)

        top = TopSettings(input_image=input_image,
                          segmentation=segmentation,
                          clip=clip,
                          postprocessing=postprocessing,
                          submit=submit)
        
        all_inputs = get_components(top)

        submit.click(
            # Python argument capture is a little weird, hence the default-argument dance.
            fn = (lambda args, top=top: processing(args, top=top)), 
            inputs=all_inputs,
            outputs=gallery
        )

    return [(PBRemTools, "PBRemTools", "pbremtools")]

script_callbacks.on_ui_tabs(on_ui_tabs)
