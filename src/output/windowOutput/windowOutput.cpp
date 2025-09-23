#include "windowOutput.hpp"
#include <chrono>

WindowOutput::WindowOutput(const std::string& windowName) :
    windowName(windowName), frameCount(0), startTime(std::chrono::high_resolution_clock::now())
{
    cv::namedWindow(windowName);
}

void WindowOutput::write(const cv::Mat& frame) 
{ 
    cv::imshow(windowName, frame); 
}

void WindowOutput::write(const cv::Mat& frame, const double& avgFps) 
{ 
    cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(avgFps)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);    
    cv::imshow(windowName, frame); 
}

void WindowOutput::close() 
{ 
    cv::destroyWindow(windowName); 
}