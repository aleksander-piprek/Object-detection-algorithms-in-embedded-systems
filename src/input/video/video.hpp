#pragma once

#ifndef INPUT_VIDEO_HPP
#define INPUT_VIDEO_HPP

#include "src/input/input.hpp"

class Video : public Input
{
    public:
        Video() = default;
        Video(const std::string& path);

        void playVideo();

        inline cv::VideoCapture& getVideo() { return cap; }
        inline cv::Mat& getFrame() { return frame; }

    private:
        cv::VideoCapture loadVideo(const std::string& path);

        cv::VideoCapture cap;
        cv::Mat frame;
};

#endif // INPUT_VIDEO_HPP