#include "imageInput.hpp"

ImageInput::ImageInput(const std::string& inputPath) :
    img(cv::imread(inputPath))
{
    if (!img.empty())
    {
        std::cout << "Image loaded successfully: " << inputPath << std::endl;
    }
    else
    {
        std::cout << "Could not open or find the image: " << inputPath << std::endl;
    }
}

bool ImageInput::read(cv::Mat& frame)
{
    if (used) 
        return false;
    frame = img.clone();
    used = true;
    return true;
}

void ImageInput::reset() 
{ 
    used = false; 
}