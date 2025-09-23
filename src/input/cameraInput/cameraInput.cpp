#include "cameraInput.hpp"

CameraInput::CameraInput(const std::string& cameraPath) :
    cap(cameraPath)
{
    if (cap.isOpened()) 
    {
        std::cout << "Camera opened successfully: " << cameraPath << std::endl;
    } else 
    {
        std::cout << "Could not open camera: " << cameraPath << std::endl;
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