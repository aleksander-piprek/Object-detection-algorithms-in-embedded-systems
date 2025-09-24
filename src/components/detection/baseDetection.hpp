#pragma once

#ifndef BASE_DETECTION_HPP
#define BASE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

struct DetectionResult
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class BaseDetection
{
    public:
        virtual ~BaseDetection() = default;    

        virtual std::vector<DetectionResult> inference(cv::Mat& frame) = 0;
        virtual void drawDetections(cv::Mat& frame, const std::vector<DetectionResult>& detections) = 0;
};

#endif // BASE_DETECTION_HPP