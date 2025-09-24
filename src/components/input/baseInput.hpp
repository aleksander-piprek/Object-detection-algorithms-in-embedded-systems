#pragma once

#ifndef BASE_INPUT_HPP
#define BASE_INPUT_HPP

#include <opencv2/opencv.hpp>

class BaseInput
{
    public:
        virtual ~BaseInput() = default;    

        virtual bool read(cv::Mat& frame) = 0;
        virtual void reset() = 0;
};

#endif // BASE_INPUT_HPP