#include <string>
#include <opencv2/opencv.hpp>

class Image
{
    cv::Mat image;
    public:
        Image(const std::string& path);
        void showImage();
    private:
        cv::Mat loadImage(const std::string& path);
};