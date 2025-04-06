#include "image.hpp"

Image::Image(const std::string& path)
{
    image = loadImage(path);
}

cv::Mat Image::loadImage(const std::string& path)
{
    image = cv::imread(path, cv::IMREAD_COLOR);
    if(image.empty())
    {
        std::cout << "Could not open or find the image: " << path << std::endl;
        return cv::Mat();
    }
    return image;
}

void Image::showImage()
{
    if (image.empty())
    {
        std::cout << "Image is empty, cannot display." << std::endl;
        return;
    }
    cv::imshow("Image", image);
    cv::imwrite("output.jpg", image);   
    cv::waitKey(0);
}