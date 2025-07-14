cmake_minimum_required(VERSION 3.15)
project(oda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BASEPATH ${CMAKE_SOURCE_DIR})

# === OpenCV ===
set(OpenCV_INSTALL_DIR ${BASEPATH}/lib/opencv/install)
find_package(OpenCV REQUIRED PATHS ${OpenCV_INSTALL_DIR} NO_DEFAULT_PATH)

# === GTest ===
set(GTest_INCLUDE_DIRS ${BASEPATH}/lib/gtest/googletest/include ${BASEPATH}/googlemock/include)
set(GTest_LIBRARIES ${BASEPATH}/lib/gtest/install/lib/libgtest.a ${BASEPATH}/lib/gtest/install/lib/libgtest_main.a)

# === ONNX Runtime ===
set(OnnxRuntime_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime")
set(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_DIR}/include)
set(OnnxRuntime_LIBRARIES ${OnnxRuntime_DIR}/lib/libonnxruntime.so)

# === cuDNN ===
set(CUDNN_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/libcudnn.so)

# set(CMAKE_CXX_CLANG_TIDY 
#     clang-tidy;
#     --extra-arg=-Wno-unknown-warning-option;
#     --extra-arg=-isystem/usr/include/c++/13;
#     --extra-arg=-isystem/usr/include/x86_64-linux-gnu/c++/13;
#     --extra-arg=-isystem/usr/include/c++/13/backward
# )

include_directories(${BASEPATH} ${CUDNN_INCLUDE_DIR})

set(SOURCES
    "main.cpp"
    "src/sandbox/sandbox.cpp"
    "src/input/image/image.cpp"
    "src/input/video/video.cpp"
    "src/detection/detection.cpp"
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

# === Build test app ===
enable_testing()
add_executable(test_oda test/test_main.cpp)
target_include_directories(test_oda PRIVATE
    ${GTest_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OnnxRuntime_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)
target_link_libraries(test_oda
    ${GTest_LIBRARIES}
    ${OnnxRuntime_LIBRARIES}
    ${CUDNN_LIBRARY}
)