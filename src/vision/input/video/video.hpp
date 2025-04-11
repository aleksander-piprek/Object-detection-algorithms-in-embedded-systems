#pragma once

#ifndef VISION_INPUT_VIDEO_HPP
#define VISION_INPUT_VIDEO_HPP

#include "src/vision/input/input.hpp"

namespace Vision
{
    class Video : public Input
    {
        public:
            Video() = default;
            Video(const std::string& path);

            void playVideo();

            inline cv::VideoCapture& getVideo() { return video; }
            inline cv::Mat& getFrame() { return frame; }

        private:
            cv::VideoCapture loadVideo(const std::string& path);

            cv::VideoCapture video;
            cv::Mat frame;
    };
}

#endif // VISION_INPUT_VIDEO_HPP