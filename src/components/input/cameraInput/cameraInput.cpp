#include "cameraInput.hpp"

CameraInput::CameraInput(const std::string& cameraPath)
{
    std::string pipeline = "v4l2src device=" + cameraPath + " ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false";
    cap = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) 
    {
        pipeline = "v4l2src device=" + cameraPath + " ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false";
        cap = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) 
        {
            throw std::runtime_error("Failed to open camera with optimized GStreamer pipeline: " + pipeline);
        }
    }
}

bool CameraInput::read(cv::Mat& frame) 
{
    return cap.read(frame);
}

void CameraInput::reset() 
{ 
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
}