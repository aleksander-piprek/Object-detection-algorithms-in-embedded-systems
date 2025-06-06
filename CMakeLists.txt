cmake_minimum_required(VERSION 3.15)
project(oda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(BASEPATH ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_CLANG_TIDY 
    clang-tidy;
    --extra-arg=-Wno-unknown-warning-option;
    --extra-arg=-isystem/usr/include/c++/13;
    --extra-arg=-isystem/usr/include/x86_64-linux-gnu/c++/13;
    --extra-arg=-isystem/usr/include/c++/13/backward
)

include_directories(${BASEPATH})

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/lib/opencv)
set(GTest_DIR ${CMAKE_SOURCE_DIR}/lib/gtest)

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR}/build)

set(GTest_INCLUDE_DIRS ${GTest_DIR}/googletest/include ${GTest_DIR}/googlemock/include)
set(GTest_LIBRARIES ${GTest_DIR}/build/lib/libgtest.a ${GTest_DIR}/build/lib/libgtest_main.a)

set(SOURCES "main.cpp"
            "src/sandbox/sandbox.cpp"
            "src/vision/window/window.cpp"
            "src/vision/input/image/image.cpp"
            "src/vision/input/video/video.cpp")

# Production build
add_executable(oda ${SOURCES})
target_include_directories(oda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(oda ${OpenCV_LIBS})

# Test build
enable_testing()
add_executable(test_oda "test/test_main.cpp")
target_include_directories(test_oda PRIVATE ${GTest_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_oda ${GTest_LIBRARIES})