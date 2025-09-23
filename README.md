# Object Detection Algorithms in Embedded Systems

My master's thesis about a modular and high-performance object detection platform using ONNX Runtime and OpenCV, designed for embedded platforms like NVIDIA Jetson. Supports image and video input with sandbox testing utilities.

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![C++](https://img.shields.io/badge/language-C++17-blue)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Tests](#tests)
- [Notes](#notes)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Features

- Plug-and-play ONNX object detection
- Supports image and video sources
- Sandbox testing environment
- Easy configuration and modular design
- Optimized mainly for NVIDIA Jetson

## Installation

### Requirements
- CMake >= 3.15
- C++17
- OpenCV
- ONNX Runtime GPU (or CPU)
- CUDA & cuDNN (for Jetson or GPU acceleration)

### Dependencies
| Name           | Version |
|----------------|---------|
| gcc            | 12.3    |
| OpenCV         | 5.x     |
| OpenCV contrib | 5.x     |
| CUDA           | 12.9    |
| cuDNN          | 9.10.2  |
| OnnxRuntime    | 1.22.0  |
 

### Clone and build
```bash
git clone --recurse-submodules https://github.com/aleksander-piprek/Object-detection-algorithms-in-embedded-systems.git
cd Object-detection-algorithms-in-embedded-systems
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Install  
```bash
./scripts/setup.sh
```

### Export YOLOV5 models
To export .onnx model you need to follow commands below
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python3 -m vevn .venv
source .venv/bin/activate
pip install -r requirements.txt
python export.py --weights yolov5s.pt --include onnx --dynamic --simplify
mkdir ~/Object-detection-algorithms-in-embedded-systems/bin
cp yolov5s.onnx ~/Object-detection-algorithms-in-embedded-systems/bin/
```

## Notes

- Do not use VS Code snap version! It uses its own packages which collides with the system's software (GLIBC, libpthread)
- Running detections on virtual machine works only on CPU (potential workaround is DDA or GPU passthrough)

## Contributing

Feel free to fork the project and open pull requests. Bug reports and suggestions are also welcome via issues.

## License

This project is licensed under the MIT License.