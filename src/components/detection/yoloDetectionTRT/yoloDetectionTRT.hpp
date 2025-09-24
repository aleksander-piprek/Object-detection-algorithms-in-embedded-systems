#pragma once

#ifndef YOLO_DETECTION_TRT_HPP
#define YOLO_DETECTION_TRT_HPP

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/components/detection/baseDetection.hpp"

class YoloDetectionTRT : public BaseDetection
{
    public:
        YoloDetectionTRT(const std::string& model_path, 
                             const std::string& class_names_path);    
        std::vector<DetectionResult> inference(cv::Mat& frame) override;
        void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections) override;

    private:
        std::vector<std::string> loadClassNames(const std::string& path);
        void* buffers[2];
        int inputIndex, outputIndex;
        int inputW, inputH;

        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        cudaStream_t stream;
        std::vector<std::string> class_names;
};

#endif // YOLO_DETECTION_TRT_HPP