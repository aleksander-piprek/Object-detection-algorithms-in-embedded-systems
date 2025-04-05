#include "window.hpp"

#include <opencv2/opencv.hpp>

Window::Window()
{
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);

    cv::putText(image, 
                "Hello, OpenCV!", 
                cv::Point(50, 100), 
                cv::FONT_HERSHEY_SIMPLEX, 
                1.0, 
                cv::Scalar(255, 255, 255), 
                2);

    cv::imshow("Image", image);

    cv::waitKey(0);
}