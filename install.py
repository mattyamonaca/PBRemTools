import launch

packages = {
    "onnx": "onnx",
    "onnxruntime": "onnxruntime-gpu==1.14.0",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
    "segmentation_refinement": "segmentation-refinement",
    "sklearn": "scikit-learn",
    "clip": "clip",
}

for name, target in packages.items():
    if not launch.is_installed(name):
        launch.run_pip(f'install {target}', desc=f'{name} for PBRemTools')

if not launch.is_installed("segment_anything"):
    launch.run_pip("install git+https://github.com/facebookresearch/segment-anything.git", desc="segment_anything for PBRemTools")
