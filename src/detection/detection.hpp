#pragma once

#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>

struct DetectionResult
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class Detection 
{
    public:
        Detection(const std::string& model_path, const std::string& class_names_path);
        
        std::vector<DetectionResult> inference(const cv::Mat& frame);
        void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections);
    
    private:
        std::vector<std::string> loadClassNames(const std::string& path);

        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;

        std::vector<std::string> input_names_str, output_names_str;
        std::vector<const char*> input_names, output_names;

        std::vector<std::string> class_names;

};  

#endif // DETECTION_HPP