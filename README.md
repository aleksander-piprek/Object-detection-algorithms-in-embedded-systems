# Object detection algorithms in embedded systems

My master's thesis about object detection. In progress...

## Dependencies
| Name           | Version |
|----------------|---------|
| gcc            | 12.3    |
| OpenCV         | 5.x     |
| OpenCV contrib | 5.x     |
| CUDA           | 12.9    |
| cuDNN          | 9.10.2  |
| OnnxRuntime    | 1.22.0  |
 
## Setup

### Cloning
This repository relies on submodules (OpenCV and GoogleTest) therefore you need to clone with --recursive flag

```bash
git clone https://github.com/aleksander-piprek/Object-detection-algorithms-in-embedded-systems.git --recursive
```

### Install  
To install and build dependencies needed to run this project please check this shell script and then run

```bash
./scripts/setup.sh
```

### Build Production
```bash
mkdir build && cd build
cmake ..
make
```

### Build Unit Tests
```bash
mkdir build && cd build
cmake ..
make test
```

### Export YOLOV5 models
To export .onnx model you need to follow commands below
```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python3 -m vevn .venv
pip install -r requirements.txt
python export.py --weights yolov5s.pt --include onnx --dynamic --simplify
mkdir ~/Object-detection-algorithms-in-embedded-systems/bin
cp yolov5s.onnx ~/Object-detection-algorithms-in-embedded-systems/bin/
```

## Running
Currently you can run it in sandbox mode for image and video inference.

## Notes

- Do not use VS Code snap version! It uses its own packages which collides with the system's software (GLIBC, libpthread)
- Running detections on virtual machine works only on CPU (potential workaround is DDA or GPU passthrough)