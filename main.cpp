#include "src/programs/example/example.hpp"
#include "src/programs/benchmark/benchmark.hpp"
#include "src/programs/videoInference/videoInference.hpp"
#include "src/programs/cameraInference/cameraInference.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    const int argumentCount = argc;

    if(argumentCount == 2)
    {
        std::string programName = argv[1];        
        
        if(programName == "--help" || programName == "-h")
        {
            std::cout << "Usage: ./oda <ProgramName>" << std::endl;
            std::cout << "Available Programs: Example, Benchmark, VideoInference, CameraInference" << std::endl;
        }
        else if(programName == "Example")
        {
            Example example;
            example.run();
        }
        else if(programName == "Benchmark")
        {    
            Benchmark benchmarkInstance;
            benchmarkInstance.run();
        }
        else if(programName == "VideoInference")
        {
            std::string videoInferenceConfigPath = "../cfg/videoInference.cfg";            
            VideoInference videoInferenceInstance(videoInferenceConfigPath);
            videoInferenceInstance.run();
        }
        else if(programName == "CameraInference")
        {
            std::string cameraInferenceConfigPath = "../cfg/cameraInference.cfg";
            CameraInference cameraInferenceInstance(cameraInferenceConfigPath);
            cameraInferenceInstance.run();
        }
        else
        {
            std::cout << "Invalid program name provided." << std::endl;
            std::cout << "Available Programs: Example, Benchmark, VideoInference, CameraInference" << std::endl;
        }
    }
    else if(argumentCount > 2)
    {
        std::cout << "Too many arguments provided." << std::endl;
        std::cout << "To see usage, use --help or -h option." << std::endl;
    }
    else
    {
        std::cout << "No arguments provided." << std::endl;
        std::cout << "To see available programs, use --help or -h option." << std::endl;
    }

    return 0;    
}