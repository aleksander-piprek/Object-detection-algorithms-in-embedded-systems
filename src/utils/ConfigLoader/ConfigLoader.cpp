#include "ConfigLoader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

ConfigLoader::ConfigLoader()
{
}

std::unordered_map<std::string, std::string> ConfigLoader::loadConfig(const std::string& configPath)
{
    std::unordered_map<std::string, std::string> configMap;
    std::ifstream in(configPath);

    if (in.is_open())
    {
        std::cout << "Loading configuration from: " << configPath << "\n\n";
    }
    else
    {
        std::cout << "Failed to open config file: " << configPath << std::endl;
        return std::unordered_map<std::string, std::string>{};
    }

    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '='))
        {
            std::string value;
            if (std::getline(iss, value))
            {
                key.erase(key.find_last_not_of(" \n\r\t") + 1);
                value.erase(0, value.find_first_not_of(" \n\r\t"));

                if (key == "ModelName")
                {
                    configMap["ModelName"] = getModelPath(value); 
                }
                else if (key == "VideoName")
                {
                    configMap["VideoName"] = getVideoPath(value);
                }
                else if (key == "CameraPath")
                {
                    configMap["CameraPath"] = value;
                }
                else if (key == "Dataset")
                {
                    configMap["Dataset"] = getClassNamesPath(value);
                }
                else if (key == "ConfThreshold")
                {
                    configMap["ConfThreshold"] = value;
                }
                else if (key == "NmsThreshold")
                {
                    configMap["NmsThreshold"] = value;
                }
                else
                {
                    std::cout << "Unknown configuration entry: " << key << std::endl;
                }
            }
        }
    }

    std::cout << "Model Path: "         << configMap["ModelName"]   << "\n";
    std::cout << "Video Path: "         << configMap["VideoName"]   << "\n";
    std::cout << "Camera Path: "        << configMap["CameraPath"]   << "\n";    
    std::cout << "Class Names Path: "       << configMap["Dataset"]     << "\n";
    std::cout << "Confidence Threshold: " << configMap["ConfThreshold"] << "\n";
    std::cout << "NMS Threshold: "      << configMap["NmsThreshold"]  << "\n\n";

    return configMap;
}

std::string ConfigLoader::getModelPath(const std::string& model_name)
{
    return "../bin/" + model_name + ".onnx";
}

std::string ConfigLoader::getVideoPath(const std::string& video_name)
{
    return "../resources/input/video/" + video_name + ".mp4";
}

std::string ConfigLoader::getClassNamesPath(const std::string& class_name)
{
    return "../bin/" + class_name + ".names";
}
