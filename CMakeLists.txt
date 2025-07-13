cmake_minimum_required(VERSION 3.15)
project(oda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(BASEPATH ${CMAKE_SOURCE_DIR})

set(OpenCV_SOURCE_DIR ${BASEPATH}/lib/opencv)
set(OpenCV_CONTRIB_DIR ${BASEPATH}/lib/opencv_contrib)
set(OpenCV_BINARY_DIR ${CMAKE_BINARY_DIR}/opencv-build)

set(GTest_DIR ${CMAKE_SOURCE_DIR}/lib/gtest)
set(OnnxRuntime_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime")

# set(CMAKE_CXX_CLANG_TIDY 
#     clang-tidy;
#     --extra-arg=-Wno-unknown-warning-option;
#     --extra-arg=-isystem/usr/include/c++/13;
#     --extra-arg=-isystem/usr/include/x86_64-linux-gnu/c++/13;
#     --extra-arg=-isystem/usr/include/c++/13/backward
# )

include_directories(${BASEPATH})

# === GTest and ONNX paths ===
set(GTest_INCLUDE_DIRS ${GTest_DIR}/googletest/include ${GTest_DIR}/googlemock/include)
set(GTest_LIBRARIES ${GTest_DIR}/build/lib/libgtest.a ${GTest_DIR}/build/lib/libgtest_main.a)

set(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_DIR}/include)
set(OnnxRuntime_LIBRARIES ${OnnxRuntime_DIR}/lib/libonnxruntime.so ${OnnxRuntime_DIR}/lib/libonnxruntime_providers_cuda.so)

set(Cuda_INCLUDE_DIRS /usr/include/cuda)
set(Cuda_LIBRARIES /usr/lib/cuda/lib64/libcudnn.so)

set(SOURCES "main.cpp"
            "src/sandbox/sandbox.cpp"
            "src/input/image/image.cpp"
            "src/input/video/video.cpp"
            "src/detection/detection.cpp")

# === Build main app ===
add_executable(oda ${SOURCES})
target_include_directories(oda PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${OnnxRuntime_INCLUDE_DIRS}
    ${Cuda_INCLUDE_DIRS}
)
target_link_libraries(oda
    ${OpenCV_LIBS}
    ${OnnxRuntime_LIBRARIES}
    ${Cuda_LIBRARIES}
)

# === Build test app ===
enable_testing()
add_executable(test_oda test/test_main.cpp)
target_include_directories(test_oda PRIVATE 
    ${GTest_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OnnxRuntime_INCLUDE_DIRS}
)
target_link_libraries(test_oda
    ${GTest_LIBRARIES}
    ${OnnxRuntime_LIBRARIES}
)