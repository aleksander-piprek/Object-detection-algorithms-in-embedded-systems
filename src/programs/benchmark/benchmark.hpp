#include <string>

class Benchmark
{
    public:
        Benchmark() = default;
        Benchmark(const std::string& configPath);  
        ~Benchmark() = default;

        void run();

    private:
        void benchmarkVideoInference();
        bool loadBenchmarkConfig(const std::string& configPath);

        std::string modelPath;
        std::string videoPath;
        std::string classNamesPath;
};