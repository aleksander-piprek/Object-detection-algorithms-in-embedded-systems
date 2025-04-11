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
        return {};
    }
    return video;
}

void Vision::Video::playVideo()
{
    while (true)
    {
        video >> frame;
        if (frame.empty())
            break;

        cv::imshow("Video", frame);
        if (cv::waitKey(30) >= 0)
            break;
    }
    video.release();
    cv::destroyAllWindows();
}