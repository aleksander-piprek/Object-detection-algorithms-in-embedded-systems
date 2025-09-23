#include "videoInference.hpp"
#include "src/input/videoInput/videoInput.hpp"
#include "src/output/windowOutput/windowOutput.hpp"
#include "src/detection/yoloDetection/yoloDetection.hpp"
#include "src/utils/ConfigLoader/ConfigLoader.hpp"
#include "src/utils/FpsCounter/FpsCounter.hpp"

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
    detection = std::make_unique<YoloDetection>(modelPath, classNamesPath, confThreshold, nmsThreshold);    

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