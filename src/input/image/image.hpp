#pragma once

#ifndef INPUT_IMAGE_HPP
#define INPUT_IMAGE_HPP

#include "src/input/input.hpp"

class Image : public Input
{
    public:
        Image() = default;
        Image(const std::string& path);

        void showImage();

        inline cv::Mat& getImage() { return image; }

    private:
        cv::Mat loadImage(const std::string& path);

        cv::Mat image;
};

#endif // INPUT_IMAGE_HPP