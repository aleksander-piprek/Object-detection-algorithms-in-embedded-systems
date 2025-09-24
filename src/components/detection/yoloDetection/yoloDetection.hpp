#pragma once

#ifndef YOLO_DETECTION_HPP
#define YOLO_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "src/components/detection/baseDetection.hpp"

class YoloDetection : public BaseDetection
{
    public:
        YoloDetection(const std::string& model_path, 
                             const std::string& class_names_path);    
        YoloDetection(const std::string& model_path, 
                             const std::string& class_names_path, 
                             const float& confThreshold, 
                             const float& nmsThreshold);
        
        std::vector<DetectionResult> inference(cv::Mat& frame) override;
        void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections) override;

    private:
        std::vector<std::string> loadClassNames(const std::string& path);

        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;

        std::vector<std::string> input_names_str, output_names_str;
        std::vector<const char*> input_names, output_names;

        std::vector<std::string> class_names;

        const int INPUT_WIDTH = 640;
        const int INPUT_HEIGHT = 640;
        float CONFIDENCE_THRESHOLD = 0.25f; // Default 
        float NMS_THRESHOLD = 0.45f;        // Default  
};  

#endif // YOLO_DETECTION_HPP