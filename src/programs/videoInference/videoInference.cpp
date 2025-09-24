#include "videoInference.hpp"
#include "src/components/input/videoInput/videoInput.hpp"
#include "src/components/output/windowOutput/windowOutput.hpp"
#include "src/components/detection/yoloDetection/yoloDetection.hpp"
#include "src/components/detection/yoloDetectionTRT/yoloDetectionTRT.hpp"
#include "src/components/utils/ConfigLoader/ConfigLoader.hpp"
#include "src/components/utils/FpsCounter/FpsCounter.hpp"

VideoInference::VideoInference(const std::string& configPath)
{
    if(!loadVideoInferenceConfig(configPath))
    {
        std::cout << "Failed to load video inference configuration." << std::endl;
        return;     
    }
}

bool VideoInference::loadVideoInferenceConfig(const std::string& configPath)
{
    auto configMap = ConfigLoader::loadConfig(configPath);
    if (configMap.empty()) 
        return false;

    modelPath = configMap["ModelName"];
    videoPath = configMap["VideoName"];
    classNamesPath = configMap["Dataset"];
    confThreshold = std::stof(configMap["ConfThreshold"]);
    nmsThreshold = std::stof(configMap["NmsThreshold"]);

    return true;
}

void VideoInference::run()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseOutput> output;
    std::unique_ptr<BaseDetection> detection;

    input = std::make_unique<VideoInput>(videoPath);
    output = std::make_unique<WindowOutput>("Video Inference");
    if(modelPath.find(".onnx") != std::string::npos)
        detection = std::make_unique<YoloDetection>(modelPath, classNamesPath, confThreshold, nmsThreshold);    
    else if(modelPath.find(".engine") != std::string::npos)
        detection = std::make_unique<YoloDetectionTRT>(modelPath, classNamesPath);
    else
    {
        std::cout << "Unsupported model format. Please use .onnx or .engine files." << std::endl;
        return;
    }

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