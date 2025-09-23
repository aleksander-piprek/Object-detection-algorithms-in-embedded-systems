#pragma once

#ifndef BASE_OUTPUT_HPP
#define BASE_OUTPUT_HPP

#include <opencv2/opencv.hpp>

class BaseOutput
{
    public:
        virtual ~BaseOutput() = default;

        virtual void write(const cv::Mat& frame) = 0;
        virtual void write(const cv::Mat& frame, const double& avgFps) = 0;

        virtual void close() = 0;
};

#endif // BASE_OUTPUT_HPP