#include "vision/window/window.hpp"
#include "vision/input/image/image.hpp"

#include <opencv2/opencv.hpp>

class Sandbox
{
    public:
        Sandbox() = default;
        void play();
        
        void imageResize();
        void imageGrayscale();
        void imageGrayscaleTransition();
};

void Sandbox::play()
{
    imageGrayscale();
}

void Sandbox::imageResize()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    Vision::Image resized;

    cv::resize(image.getImage(), resized.getImage(), cv::Size(1280, 600));
    resized.showImage();
}

void Sandbox::imageGrayscale()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    Vision::Image gray;

    cv::cvtColor(image.getImage(), gray.getImage(), cv::COLOR_BGR2GRAY);
    gray.showImage();
}

void Sandbox::imageGrayscaleTransition()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    Vision::Image transitions;

    for(auto& colors : colorCodes)
    {
        cv::cvtColor(image.getImage(), transitions.getImage(), colors);
        transitions.showImage();
    }
}