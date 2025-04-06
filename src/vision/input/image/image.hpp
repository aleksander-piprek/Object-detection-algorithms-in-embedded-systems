#include <string>
#include <opencv2/opencv.hpp>

#include "../input.hpp"

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