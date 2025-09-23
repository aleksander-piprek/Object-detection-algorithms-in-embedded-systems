#include "benchmark.hpp"
#include "src/input/imageInput/imageInput.hpp"
#include "src/input/videoInput/videoInput.hpp"
#include "src/input/cameraInput/cameraInput.hpp"
#include "src/output/windowOutput/windowOutput.hpp"
#include "src/detection/yoloDetection/yoloDetection.hpp"
#include "src/utils/ConfigLoader/ConfigLoader.hpp"

#include <numeric>

Benchmark::Benchmark(const std::string& configPath)
{
    if(!loadBenchmarkConfig(configPath))
    {
        std::cout << "Failed to load benchmark configuration." << std::endl;
        return;     
    }
}

void Benchmark::run()
{
    benchmarkVideoInference();
}

bool Benchmark::loadBenchmarkConfig(const std::string& configPath)
{
    auto configMap = ConfigLoader::loadConfig(configPath);
    if (configMap.empty()) 
        return false;

    modelPath = configMap["ModelName"];
    videoPath = configMap["VideoName"];
    classNamesPath = configMap["Dataset"];

    return true; 
}

void Benchmark::benchmarkVideoInference()
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseDetection> detection;

    input = std::make_unique<VideoInput>(videoPath);
    detection = std::make_unique<YoloDetection>(modelPath, classNamesPath);    

    cv::Mat frame;
    input->read(frame);
    for (int i = 0; i < 5; i++) 
    {
        detection->inference(frame);
    }

    int frameCount = 0;
    int totalDetections = 0;
    double totalScore = 0.0;
    std::vector<double> latencies;

    auto start = std::chrono::high_resolution_clock::now();
    
    while (input->read(frame)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto detections = detection->inference(frame);
        auto t1 = std::chrono::high_resolution_clock::now();

        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        frameCount++;
        totalDetections += detections.size();
        for (const auto& det : detections) totalScore += det.confidence;
    }

    auto end = std::chrono::high_resolution_clock::now();

    double elapsedSec = std::chrono::duration<double>(end - start).count();
    double avgFPS = frameCount / elapsedSec;
    double avgLatency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double avgScore = totalDetections > 0 ? totalScore / totalDetections : 0.0;

    std::cout << "Frames: " << frameCount << "\n";
    std::cout << "Processing FPS: " << avgFPS << "\n";
    std::cout << "Avg latency (ms): " << avgLatency << "\n";
    std::cout << "Total detections: " << totalDetections << "\n";
    std::cout << "Avg detection confidence: " << avgScore << "\n";
}
