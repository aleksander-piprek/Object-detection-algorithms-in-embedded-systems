#pragma once

#ifndef VISION_INPUT_IMAGE_HPP
#define VISION_INPUT_IMAGE_HPP

#include "src/vision/input/input.hpp"

namespace Vision
{
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
}

#endif // VISION_INPUT_IMAGE_HPP