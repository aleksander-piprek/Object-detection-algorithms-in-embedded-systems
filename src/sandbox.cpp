#include "vision/window/window.hpp"
#include "vision/input/image/image.hpp"

#include <opencv2/opencv.hpp>

class Sandbox
{
    public:
        Sandbox() = default;
        void play();
};

void Sandbox::play()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    Vision::Image resized;

    cv::resize(image.getImage(), resized.getImage(), cv::Size(1280, 600));
    resized.showImage();
}