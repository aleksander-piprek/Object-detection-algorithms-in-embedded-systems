cmake_minimum_required(VERSION 3.15)
project(oda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/lib/opencv)

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR}/build)

set(SOURCES "main.cpp"
            "src/window/window.cpp")

add_executable(oda ${SOURCES})

target_include_directories(oda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(oda ${OpenCV_LIBS})