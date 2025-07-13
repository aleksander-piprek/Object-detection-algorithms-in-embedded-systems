#pragma once

#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/opencv.hpp>

int inference(const std::string& inputPath);

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

#endif // DETECTION_HPP