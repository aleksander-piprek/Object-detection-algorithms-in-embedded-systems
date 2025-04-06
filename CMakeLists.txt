cmake_minimum_required(VERSION 3.15)
project(oda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory("lib/gtest")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/lib/opencv)
set(GTest_DIR ${CMAKE_SOURCE_DIR}/lib/gtest)

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR}/build)

enable_testing()

set(SOURCES "main.cpp"
            "src/window/window.cpp"
            "src/image/image.cpp")

add_executable(oda ${SOURCES})

target_include_directories(oda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(oda ${OpenCV_LIBS})

add_executable(test_oda "test/test_main.cpp")
target_link_libraries(test_oda gtest gtest_main)
add_test(NAME TestOda COMMAND test_oda)