import launch

if not launch.is_installed("onnx"):
    launch.run_pip("install onnx", "Installing onnx...")

if not launch.is_installed("onnxruntime-gpu"):
    launch.run_pip("install onnxruntime-gpu", "Installing onnxruntime-gpu...")

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "Installing opencv-python...")

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", "Installing numpy...")

if not launch.is_installed("Pillow"):
    launch.run_pip("install Pillow", "Installing Pillow...")

if not launch.is_installed("segmentation-refinement"):
    launch.run_pip("install segmentation-refinement", "Installing segmentation-refinement...")

if not launch.is_installed("scikit-learn"):
    launch.run_pip("install scikit-learn", "Installing scikit-learn...")

if not launch.is_installed("clip"):
    launch.run_pip("install clip", "Installing clip...")

if not launch.is_installed("segment_anything"):
    launch.run_pip("install git+https://github.com/facebookresearch/segment-anything.git", "Installing segment_anything...")