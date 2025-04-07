#include <string>
#include <opencv2/opencv.hpp>

#include "src/vision/input/input.hpp"

namespace Vision
{
    class Video : public Input
    {
        cv::VideoCapture video;
        cv::Mat frame;

        public:
            Video(const std::string& path);
            void showVideo();
        private:
            cv::VideoCapture loadVideo(const std::string& path);
    };
}