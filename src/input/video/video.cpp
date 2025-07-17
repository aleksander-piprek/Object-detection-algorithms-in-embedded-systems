#include "video.hpp"

Video::Video(const std::string& path)
{
    cap = loadVideo(path);
}

cv::VideoCapture Video::loadVideo(const std::string& path)
{
    cap.open(path);
    if (!cap.isOpened())
    {
        std::cout << "Could not open or find the video: " << path << std::endl;
        return {};
    }
    return cap;
}

void Video::playVideo()
{
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    int delay = (video_fps > 0) ? static_cast<int>(1000.0 / video_fps) : 33;

    while (cap.read(frame))
    {
        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(video_fps)),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 1);
        cv::imshow("Video", frame);
        int key = cv::waitKey(delay);
        if (key == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
}