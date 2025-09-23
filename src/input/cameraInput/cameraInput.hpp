#pragma once

#ifndef CAMERA_INPUT_HPP
#define CAMERA_INPUT_HPP

#include "src/input/baseInput.hpp" 

class CameraInput : public BaseInput
{
    public:
        CameraInput(const std::string& cameraPath);

        bool read(cv::Mat& frame) override;
        void reset() override;

    private:
        cv::VideoCapture cap;
};

#endif // CAMERA_INPUT_HPP