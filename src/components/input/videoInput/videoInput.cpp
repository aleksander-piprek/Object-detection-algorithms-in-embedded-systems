#include "videoInput.hpp"

VideoInput::VideoInput(const std::string& inputPath) :
    cap(inputPath)
{
    if (cap.isOpened())
    {
        std::cout << "Video opened successfully: " << inputPath << std::endl;
    }
    else
    {
        std::cout << "Could not open or find the video: " << inputPath << std::endl;
    }
}

bool VideoInput::read(cv::Mat& frame)
{ 
    return cap.read(frame); 
}

void VideoInput::reset() 
{ 
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
}