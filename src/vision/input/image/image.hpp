#include <string>
#include <opencv2/opencv.hpp>

#include "../input.hpp"

namespace Vision
{
    class Image : public Input
    {
        cv::Mat image;
        public:
            Image(const std::string& path);
            void showImage();
        private:
            cv::Mat loadImage(const std::string& path);
    };
}