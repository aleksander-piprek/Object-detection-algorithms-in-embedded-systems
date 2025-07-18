#include "sandbox.hpp"
#include "src/input/imageInput/imageInput.hpp"
#include "src/input/videoInput/videoInput.hpp"

#include "src/output/windowOutput/windowOutput.hpp"

#include "src/detection/detection.hpp"

void Sandbox::play()
{
    // // 
    // // Image Processing Examples
    // //
    // imageResize();
    // imageGrayscale();
    // imageBlur();
    // imageEdgeDetection();
    // rotateImage();
    // imageInference();
    
    // // 
    // // Video Processing Example
    // //
    // playVideo();
    videoInference();
}

void Sandbox::imageResize()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    
    input = std::make_unique<ImageInput>("../resources/input/image/Wagtail.jpg");
    output = std::make_unique<WindowOutput>("Resized Image");
    cv::Mat frame;

    input->read(frame);
    cv::resize(frame, frame, cv::Size(1280, 600));

    while(true)
    {
        output->write(frame);
        if (cv::waitKey(30) >= 0) 
            break;
    }
}

void Sandbox::imageGrayscale()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    
    input = std::make_unique<ImageInput>("../resources/input/image/Wagtail.jpg");
    output = std::make_unique<WindowOutput>("Grayscale Image");
    cv::Mat frame;

    input->read(frame);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);    
    
    while(true)
    {
        output->write(frame);
        if (cv::waitKey(30) >= 0) 
            break;
    }
}

void Sandbox::imageBlur()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    
    input = std::make_unique<ImageInput>("../resources/input/image/Wagtail.jpg");
    output = std::make_unique<WindowOutput>("Grayscale Image");
    cv::Mat frame;

    input->read(frame);
    cv::GaussianBlur(frame, frame, cv::Size(9, 9), 0);    
    
    while(true)
    {
        output->write(frame);
        if (cv::waitKey(30) >= 0) 
            break;
    }
}

void Sandbox::imageEdgeDetection()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    
    input = std::make_unique<ImageInput>("../resources/input/image/Wagtail.jpg");
    output = std::make_unique<WindowOutput>("Grayscale Image");
    cv::Mat frame;
    cv::Mat edges;

    input->read(frame);
    cv::Canny(frame, edges, 100, 200);
    
    while(true)
    {
        output->write(edges);
        if (cv::waitKey(30) >= 0) 
            break;
    }    
}

void Sandbox::rotateImage()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    
    input = std::make_unique<ImageInput>("../resources/input/image/Wagtail.jpg");
    output = std::make_unique<WindowOutput>("Grayscale Image");
    cv::Mat frame;

    input->read(frame);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(frame.cols / 2, frame.rows / 2), 45, 1);
    cv::warpAffine(frame, frame, rotationMatrix, frame.size());

    while(true)
    {
        output->write(frame);
        if (cv::waitKey(30) >= 0) 
            break;
    }
}

void Sandbox::imageInference()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    std::unique_ptr<Detection> detection;

    input = std::make_unique<ImageInput>("../resources/input/image/Wagtail.jpg");
    output = std::make_unique<WindowOutput>("Image Inference");
    detection = std::make_unique<Detection>("../bin/yolov5s.onnx", "../resources/coco.names");    
    cv::Mat frame;

    input->read(frame);
    auto detections = detection->inference(frame);
    detection->drawDetections(frame, detections);

    while (true) 
    {
        output->write(frame);
        if (cv::waitKey(30) >= 0) 
            break;
    }
}

void Sandbox::playVideo()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;

    input = std::make_unique<VideoInput>("../resources/input/video/people-detection.mp4");
    output = std::make_unique<WindowOutput>("Video Playback");
    cv::Mat frame;

    while(input->read(frame))
    {
        output->write(frame);
        if (cv::waitKey(30) >= 0) 
            break;
    }
}

void Sandbox::videoInference()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    std::unique_ptr<Detection> detection;

    input = std::make_unique<VideoInput>("../resources/input/video/people-detection.mp4");
    output = std::make_unique<WindowOutput>("Video Inference");
    detection = std::make_unique<Detection>("../bin/yolov5s.onnx", "../resources/coco.names");    
    cv::Mat frame;

    while (input->read(frame)) 
    {
        auto detections = detection->inference(frame);
        detection->drawDetections(frame, detections);        
        output->write(frame);
        if (cv::waitKey(30) == 0) 
            break;
    }
}