#pragma once

#ifndef YOLO_DETECTION_TRT_HPP
#define YOLO_DETECTION_TRT_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <memory>

#include "src/components/detection/baseDetection.hpp"

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};

class YoloDetectionTRT : public BaseDetection {
public:
    YoloDetectionTRT(const std::string& engine_path, const std::string& class_names_path);
    YoloDetectionTRT(const std::string& engine_path, const std::string& class_names_path,
                     const float& confThreshold, const float& nmsThreshold);
    ~YoloDetectionTRT();

    std::vector<DetectionResult> inference(cv::Mat& frame) override;
    void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections) override;

private:
    std::vector<std::string> loadClassNames(const std::string& path);

    Logger logger;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::vector<void*> buffers;
    std::vector<std::string> binding_names;
    std::vector<bool> is_input;
    std::vector<nvinfer1::Dims> binding_dims;
    cudaStream_t stream;

    std::vector<std::string> class_names;
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    float CONFIDENCE_THRESHOLD = 0.25f; // Default
    float NMS_THRESHOLD = 0.45f;        // Default
};

#endif // YOLO_DETECTION_TRT_HPP