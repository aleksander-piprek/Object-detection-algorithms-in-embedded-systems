#pragma once

#ifndef IMAGE_INPUT_HPP
#define IMAGE_INPUT_HPP

#include "src/input/baseInput.hpp"

class ImageInput : public BaseInput
{
    public:
        ImageInput(const std::string& inputPath);

        bool read(cv::Mat& frame) override; 
        void reset() override;

    private:
        cv::Mat img;
        bool used = false;
};

#endif // INPUT_IMAGE_HPP