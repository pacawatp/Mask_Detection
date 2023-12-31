# Mask_Detection
Use the yolov5 model to detect faces and classify masked, non-masked and bad-masked.

Run in terminal
>>> cd ...\mask\yolo\yolov5-master  \
>>> python .\mask_detection.py  \
press 'q' to close

# Recommend
Plotting --------------------------------------------------------------------\
pandas>=1.1.4\
seaborn>=0.11.0\
Export ----------------------------------------------------------------------\
coremltools>=6.0  # CoreML export\
onnx>=1.12.0  # ONNX export\
onnx-simplifier>=0.4.1  # ONNX simplifier\
nvidia-pyindex  # TensorRT export\
nvidia-tensorrt  # TensorRT export\
scikit-learn<=1.1.2  # CoreML quantization\
tensorflow>=2.4.1  # TF exports (-cpu, -aarch64, -macos)\
tensorflowjs>=3.9.0  # TF.js export\
openvino-dev  # OpenVINO export\
Deploy ----------------------------------------------------------------------\
setuptools>=65.5.1 # Snyk vulnerability fix\
wheel>=0.38.0 # Snyk vulnerability fix\

tritonclient[all]~=2.24.0\
Extras ----------------------------------------------------------------------\
mss  screenshots\
albumentations>=1.0.3\
pycocotools>=2.0.6  # COCO mAP\
roboflow\
ultralytics  # HUB https://hub.ultralytics.com
