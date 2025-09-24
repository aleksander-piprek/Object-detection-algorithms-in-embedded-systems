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

# === cuDNN ===
set(CUDNN_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/libcudnn.so)

include_directories(${BASEPATH} ${CUDNN_INCLUDE_DIR})

set(SOURCES
    "main.cpp"
    ### Programs
    "src/programs/example/example.cpp"
    "src/programs/benchmark/benchmark.cpp"
    "src/programs/videoInference/videoInference.cpp"
    "src/programs/cameraInference/cameraInference.cpp"
    ### Components
    "src/input/imageInput/imageInput.cpp"
    "src/input/videoInput/videoInput.cpp"
    "src/input/cameraInput/cameraInput.cpp"
    "src/detection/yoloDetection/yoloDetection.cpp"
    "src/output/windowOutput/windowOutput.cpp"
    ### Utils
    "src/utils/FpsCounter/FpsCounter.cpp"
    "src/utils/ConfigLoader/ConfigLoader.cpp"
)

# === Build main app ===
add_executable(oda ${SOURCES})
target_include_directories(oda PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${OnnxRuntime_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)
target_link_libraries(oda
    ${OpenCV_LIBS}
    ${OnnxRuntime_LIBRARIES}
    ${CUDNN_LIBRARY}
)