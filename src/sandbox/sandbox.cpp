#include "sandbox.hpp"
#include "src/vision/input/image/image.hpp"
#include "src/vision/input/video/video.hpp"

void Sandbox::play()
{
    // imageResize();
    // imageGrayscale();
    // imageBlur();
    // imageEdgeDetection();
    // rotateImage();
    
    playVideo();
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

void Sandbox::imageBlur()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    Vision::Image blurredImage;

    cv::GaussianBlur(image.getImage(), blurredImage.getImage(), cv::Size(9, 9), 0);
    blurredImage.showImage();
}

void Sandbox::imageEdgeDetection()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    Vision::Image imageEdges;

    cv::Canny(image.getImage(), imageEdges.getImage(), 100, 200);
    imageEdges.showImage();
}

void Sandbox::rotateImage()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    Vision::Image rotatedImage;

    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(image.getImage().cols / 2, image.getImage().rows / 2), 45, 1);
    cv::warpAffine(image.getImage(), rotatedImage.getImage(), rotationMatrix, image.getImage().size());
    rotatedImage.showImage();
}

void Sandbox::playVideo()
{
    Vision::Video video("../resources/videos/test.mp4");
    video.playVideo();
}