import launch

packages = [
    "onnx",
    "onnxruntime-gpu",
    "opencv-python",
    "numpy",
    "Pillow",
    "segmentation-refinement",
    "scikit-learn",
    "clip",
]

for package in packages:
    if not launch.is_installed(package):
        launch.run_pip(f'install {package}', desc=f'{package} for PBRemTools')

if not launch.is_installed("segment_anything"):
    launch.run_pip("install git+https://github.com/facebookresearch/segment-anything.git", desc="segment_anything")
