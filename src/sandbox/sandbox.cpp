#include "sandbox.hpp"
#include "src/input/image/image.hpp"
#include "src/input/video/video.hpp"
#include "src/detection/detection.hpp"

void Sandbox::play()
{
    // // 
    // // Image Processing Examples
    // // 
    // imageResize();
    // imageGrayscale();
    // imageBlur();
    // imageEdgeDetection();
    // rotateImage();
    imageProcessDetections();
    
    // // 
    // // Video Processing Example
    // //
    // playVideo();

}

void Sandbox::imageResize()
{
    Image image("../resources/images/Wagtail.JPG");
    Image resized;

    cv::resize(image.getImage(), resized.getImage(), cv::Size(1280, 600));
    resized.showImage();
}

void Sandbox::imageGrayscale()
{
    Image image("../resources/images/Wagtail.JPG");
    Image gray;

    cv::cvtColor(image.getImage(), gray.getImage(), cv::COLOR_BGR2GRAY);
    gray.showImage();
}

void Sandbox::imageBlur()
{
    Image image("../resources/images/Wagtail.JPG");
    Image blurredImage;

    cv::GaussianBlur(image.getImage(), blurredImage.getImage(), cv::Size(9, 9), 0);
    blurredImage.showImage();
}

void Sandbox::imageEdgeDetection()
{
    Image image("../resources/images/Wagtail.JPG");
    Image imageEdges;

    cv::Canny(image.getImage(), imageEdges.getImage(), 100, 200);
    imageEdges.showImage();
}

void Sandbox::rotateImage()
{
    Image image("../resources/images/Wagtail.JPG");
    Image rotatedImage;

    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(image.getImage().cols / 2, image.getImage().rows / 2), 45, 1);
    cv::warpAffine(image.getImage(), rotatedImage.getImage(), rotationMatrix, image.getImage().size());
    rotatedImage.showImage();
}

void Sandbox::playVideo()
{
    Video video("../resources/videos/test.mp4");
    video.playVideo();
}

void Sandbox::imageProcessDetections()
{
    inference();
}