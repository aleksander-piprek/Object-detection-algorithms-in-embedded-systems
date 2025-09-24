#include "example.hpp"
#include "src/components/input/imageInput/imageInput.hpp"
#include "src/components/input/videoInput/videoInput.hpp"
#include "src/components/input/cameraInput/cameraInput.hpp"
#include "src/components/output/windowOutput/windowOutput.hpp"
#include "src/components/detection/yoloDetection/yoloDetection.hpp"
#include "src/components/utils/FpsCounter/FpsCounter.hpp"

void Example::run()
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

    // // 
    // // Camera Processing Example
    // //
    // playCamera();
    // cameraInference();
}

// // 
// // Image Processing Examples
// //

void Example::imageResize()
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
        if (cv::waitKey(1) == 27) 
            break;
    }
}

void Example::imageGrayscale()
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
        if (cv::waitKey(1) == 27) 
            break;
    }
}

void Example::imageBlur()
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
        if (cv::waitKey(1) == 27) 
            break;
    }
}

void Example::imageEdgeDetection()
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
        if (cv::waitKey(1) == 27) 
            break;
    }    
}

void Example::rotateImage()
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
        if (cv::waitKey(1) == 27) 
            break;
    }
}

void Example::imageInference()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    std::unique_ptr<BaseDetection> detection;

    input = std::make_unique<ImageInput>("../resources/input/image/Crow.jpg");
    output = std::make_unique<WindowOutput>("Image Inference");
    detection = std::make_unique<YoloDetection>("../bin/yolov5s.onnx", "../resources/coco.names");    
    cv::Mat frame;

    input->read(frame);
    auto detections = detection->inference(frame);
    detection->drawDetections(frame, detections);
    cv::imwrite("../resources/output/image/jpg.jpg", frame);
    while (true) 
    {
        output->write(frame);
        if (cv::waitKey(1) == 27) 
            break;
    }
}

// // 
// // Video Processing Example
// //

void Example::playVideo()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;

    input = std::make_unique<VideoInput>("../resources/input/video/people-detection.mp4");
    output = std::make_unique<WindowOutput>("Video Playback");
    cv::Mat frame;

    while(input->read(frame))
    {
        output->write(frame);
        if (cv::waitKey(1) == 27) 
            break;
    }
}

void Example::videoInference()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    std::unique_ptr<BaseDetection> detection;

    input = std::make_unique<VideoInput>("../resources/input/video/video1.mp4");
    output = std::make_unique<WindowOutput>("Video Inference");
    detection = std::make_unique<YoloDetection>("../bin/yolov5m.onnx", "../resources/coco.names");    

    cv::Mat frame;
    FpsCounter fpsCounter;

    while (input->read(frame)) 
    {        
        double avg_fps = fpsCounter.update();
        
        auto detections = detection->inference(frame);
        detection->drawDetections(frame, detections);        

        output->write(frame, avg_fps);

        if (cv::waitKey(1) == 27) 
            break;
    }
}

// // 
// // Camera Processing Example
// //

void Example::playCamera()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;

    input = std::make_unique<CameraInput>("ls/dev/");
    output = std::make_unique<WindowOutput>("Camera Playback");
    cv::Mat frame;

    while(input->read(frame))
    {
        output->write(frame);
        if (cv::waitKey(1) == 27) 
            break;
    }
}

void Example::cameraInference()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    std::unique_ptr<BaseDetection> detection;

    input = std::make_unique<CameraInput>("0"); 
    output = std::make_unique<WindowOutput>("Camera Inference");
    detection = std::make_unique<YoloDetection>("../bin/yolov5s.onnx", "../resources/coco.names");    
    cv::Mat frame;

    while (input->read(frame)) 
    {
        auto detections = detection->inference(frame);
        detection->drawDetections(frame, detections);        
        output->write(frame);
        if (cv::waitKey(1) == 27) 
            break;
    }
}