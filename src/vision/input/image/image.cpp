#include "image.hpp"

Vision::Image::Image(const std::string& path)
{
    image = loadImage(path);
}

cv::Mat Vision::Image::loadImage(const std::string& path)
{
    image = cv::imread(path, cv::IMREAD_COLOR);
    if(image.empty())
    {
        std::cout << "Could not open or find the image: " << path << std::endl;
        return {};
    }
    return image;
}

void Vision::Image::showImage()
{
    if (image.empty())
    {
        std::cout << "Image is empty, cannot display." << std::endl;
        return;
    }
    cv::imshow("Image", image);
    cv::waitKey(0);
}