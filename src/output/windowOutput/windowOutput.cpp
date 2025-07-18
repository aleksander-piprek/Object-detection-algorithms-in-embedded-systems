#include "windowOutput.hpp"

WindowOutput::WindowOutput(const std::string& windowName) :
    windowName(windowName)
{
    cv::namedWindow(windowName);
}

void WindowOutput::write(const cv::Mat& frame) 
{ 
    cv::imshow(windowName, frame); 
}

void WindowOutput::close() 
{ 
    cv::destroyWindow(windowName); 
}