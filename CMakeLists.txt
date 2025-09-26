cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER /usr/bin/x86_64-linux-gnu-gcc-12)
set(CMAKE_CXX_COMPILER /usr/bin/x86_64-linux-gnu-g++-12)

project(oda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BASEPATH ${CMAKE_SOURCE_DIR})

# === OpenCV ===
set(OpenCV_INSTALL_DIR ${BASEPATH}/lib/opencv/install)
set(OpenCV_INCLUDE_DIRS ${OpenCV_INSTALL_DIR}/include/opencv4)
set(OpenCV_LIBRARIES
    ${OpenCV_INSTALL_DIR}/lib/libopencv_core.so
    ${OpenCV_INSTALL_DIR}/lib/libopencv_imgproc.so
    ${OpenCV_INSTALL_DIR}/lib/libopencv_highgui.so
    ${OpenCV_INSTALL_DIR}/lib/libopencv_imgcodecs.so
    ${OpenCV_INSTALL_DIR}/lib/libopencv_videoio.so
    ${OpenCV_INSTALL_DIR}/lib/libopencv_dnn.so
)

# === ONNX Runtime ===
set(OnnxRuntime_INCLUDE_DIRS /usr/local/include/onnxruntime)
set(OnnxRuntime_LIBRARIES /usr/local/lib/libonnxruntime.so)

# === CUDA ===
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
find_library(CUDA_CUDART_LIBRARY cudart HINTS /usr/local/cuda/lib64)
find_library(CUDA_CUBLAS_LIBRARY cublas HINTS /usr/local/cuda/lib64)
set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARY})

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

target_link_libraries(
    oda
    ${OpenCV_LIBRARIES}
    ${OnnxRuntime_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARY}
    ${CUDNN_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
)