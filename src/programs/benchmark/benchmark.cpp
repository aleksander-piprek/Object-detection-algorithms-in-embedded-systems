#include "benchmark.hpp"
#include "src/components/input/imageInput/imageInput.hpp"
#include "src/components/input/videoInput/videoInput.hpp"
#include "src/components/input/cameraInput/cameraInput.hpp"
#include "src/components/output/windowOutput/windowOutput.hpp"
#include "src/components/detection/yoloDetection/yoloDetection.hpp"
#include "src/components/detection/yoloDetectionTRT/yoloDetectionTRT.hpp"
#include "src/components/utils/ConfigLoader/ConfigLoader.hpp"

#include <numeric>
#include <fstream>

void Benchmark::run()
{
    for(const auto& modelPath : models)
    {
        for(const auto& videoPath : videosPaths)
        {
            std::cout << "Benchmarking Model: " << modelPath << " on Video: " << videoPath << std::endl;
            benchmarkVideoInference(videoPath, modelPath);
        }
    }
}

void Benchmark::benchmarkVideoInference(std::string videoPath, std::string modelPath)
{
    std::unique_ptr<BaseInput> input;
    std::unique_ptr<BaseDetection> detection;

    const std::string video_path = "../resources/input/video/" + videoPath;
    const std::string model_path = "../bin/" + modelPath;

    input = std::make_unique<VideoInput>(video_path);
    if(model_path.find(".onnx") != std::string::npos)
        detection = std::make_unique<YoloDetection>(model_path, "..bin/coco.names");    
    else if(model_path.find(".engine") != std::string::npos)
        detection = std::make_unique<YoloDetectionTRT>(model_path, "..bin/coco.names");
    else
    {
        std::cout << "Unsupported model format. Please use .onnx or .engine files." << std::endl;
        return;
    }

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

    std::ofstream resultFile("../logs/benchmark_results.log", std::ios::app);
    if (resultFile.is_open()) 
    {
        resultFile << "Model: " << modelPath << ", Video: " << videoPath << "\n";
        resultFile << "Frames: " << frameCount << "\n";
        resultFile << "Processing FPS: " << avgFPS << "\n";
        resultFile << "Avg latency (ms): " << avgLatency << "\n";
        resultFile << "Total detections: " << totalDetections << "\n";
        resultFile << "Avg detection confidence: " << avgScore << "\n";
        resultFile << "-----------------------------\n";
        resultFile.close();
    } 
    else 
    {
        std::cout << "Failed to open benchmark_results.txt for writing.\n";
    }
}