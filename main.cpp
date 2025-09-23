#include "src/programs/example/example.hpp"
#include "src/programs/benchmark/benchmark.hpp"
#include "src/programs/videoInference/videoInference.hpp"

#include <string>

int main()
{
    std::string configPath = "../cfg/videoInference.cfg";

    // Example example;
    // example.run();

    // Benchmark benchmarkInstance(configPath);
    // benchmarkInstance.run();

    VideoInference videoInferenceInstance(configPath);
    videoInferenceInstance.run();

    return 0;    
}