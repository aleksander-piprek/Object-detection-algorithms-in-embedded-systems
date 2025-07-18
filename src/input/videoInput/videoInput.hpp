#pragma once

#ifndef VIDEO_INPUT_HPP
#define VIDEO_INPUT_HPP

#include "src/input/baseInput.hpp" 

class VideoInput : public BaseInput
{
    public:
        VideoInput(const std::string& inputPath);

        bool read(cv::Mat& frame) override;
        void reset() override;

    private:
        cv::VideoCapture cap;
};

#endif // VIDEO_INPUT_HPP