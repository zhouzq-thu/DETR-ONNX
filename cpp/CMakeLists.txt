cmake_minimum_required(VERSION 3.18)
project(DETR_ONNX CXX)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED core imgproc highgui)

# Find ONNXRuntime
find_package(onnxruntime REQUIRED)

add_executable(detr_onnx main.cpp)
target_link_libraries(detr_onnx ${OpenCV_LIBS} onnxruntime::onnxruntime)
