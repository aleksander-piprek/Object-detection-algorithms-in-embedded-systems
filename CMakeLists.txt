cmake_minimum_required(VERSION 3.15)
project(oda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BASEPATH ${CMAKE_SOURCE_DIR})

# === OpenCV ===
set(OpenCV_INSTALL_DIR ${BASEPATH}/lib/opencv/install)
find_package(OpenCV REQUIRED PATHS ${OpenCV_INSTALL_DIR} NO_DEFAULT_PATH)

# === ONNX Runtime ===
set(OnnxRuntime_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime")
set(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_DIR}/include)
set(OnnxRuntime_LIBRARIES ${OnnxRuntime_DIR}/lib/libonnxruntime.so)

# === CUDA ===
find_package(CUDA REQUIRED)
find_library(CUDART_LIBRARY cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# === cuDNN ===
set(CUDNN_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
find_library(CUDNN_LIBRARY cudnn HINTS /usr/lib/x86_64-linux-gnu)

# === TensorRT ===
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/x86_64-linux-gnu)

include_directories(
    ${BASEPATH}
    ${OpenCV_INCLUDE_DIRS}
    ${OnnxRuntime_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
)

set(SOURCES
    "main.cpp"
    ### Programs
    "src/programs/example/example.cpp"
    "src/programs/benchmark/benchmark.cpp"
    "src/programs/videoInference/videoInference.cpp"
    "src/programs/cameraInference/cameraInference.cpp"
    ### Components
    "src/components/input/imageInput/imageInput.cpp"
    "src/components/input/videoInput/videoInput.cpp"
    "src/components/input/cameraInput/cameraInput.cpp"
    "src/components/detection/yoloDetection/yoloDetection.cpp"
    "src/components/detection/yoloDetectionTRT/yoloDetectionTRT.cpp"    
    "src/components/output/windowOutput/windowOutput.cpp"
    ### Utils
    "src/components/utils/FpsCounter/FpsCounter.cpp"
    "src/components/utils/ConfigLoader/ConfigLoader.cpp"
)

# === Build main app ===
add_executable(oda ${SOURCES})
target_link_libraries(oda
    ${OpenCV_LIBS}
    ${OnnxRuntime_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARY}
    ${CUDNN_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
)