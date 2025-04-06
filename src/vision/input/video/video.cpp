#include "video.hpp"

Vision::Video::Video(const std::string& path)
{
    video = loadVideo(path);
}

cv::VideoCapture Vision::Video::loadVideo(const std::string& path)
{
    video.open(path);
    if (!video.isOpened())
    {
        std::cout << "Could not open or find the video: " << path << std::endl;
        return cv::VideoCapture();
    }
    return video;
}

void Vision::Video::showVideo()
{
    
}