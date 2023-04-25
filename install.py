import launch
import sys

if not launch.is_installed("onnx"):
    launch.run_pip("install onnx", "onnx...")

if not launch.is_installed("onnxruntime-gpu"):
    launch.run_pip("install onnxruntime-gpu", "onnxruntime-gpu...")

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "opencv-python...")

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", "numpy...")

if not launch.is_installed("Pillow"):
    launch.run_pip("install Pillow", "Pillow...")

if not launch.is_installed("segmentation-refinement"):
    launch.run_pip("install segmentation-refinement", "segmentation-refinement...")

if not launch.is_installed("scikit-learn"):
    launch.run_pip("install scikit-learn", "scikit-learn...")

if not launch.is_installed("clip"):
    launch.run_pip("install clip", "clip...")

if not launch.is_installed("segment_anything"):
    launch.run_pip("install git+https://github.com/facebookresearch/segment-anything.git", "segment_anything...")

if not launch.is_installed("nptyping"):
    launch.run_pip("install nptyping", "nptyping...")

if sys.version_info < (3, 9):
    # Native in 3.9 and above.
    if not launch.is_installed("typing_extensions"):
        launch.run_pip("install typing_extensions", "typing_extensions...")
