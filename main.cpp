#include "src/programs/example/example.hpp"
#include "src/programs/benchmark/benchmark.hpp"
#include "src/programs/videoInference/videoInference.hpp"
#include "src/programs/cameraInference/cameraInference.hpp"

#include <string>

int main()
{
    std::string videoInferenceConfigPath = "../cfg/videoInference.cfg";
    std::string cameraInferenceConfigPath = "../cfg/cameraInference.cfg";

    // Example example;
    // example.run();

    // Benchmark benchmarkInstance(configPath);
    // benchmarkInstance.run();

    // VideoInference videoInferenceInstance(videoInferenceConfigPath);
    // videoInferenceInstance.run();

    CameraInference cameraInferenceInstance(cameraInferenceConfigPath);
    cameraInferenceInstance.run();

    return 0;    
}