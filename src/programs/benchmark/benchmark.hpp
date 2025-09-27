#include <string>

class Benchmark
{
    public:
        Benchmark() = default;
        ~Benchmark() = default;

        void run();

    private:
        void benchmarkVideoInference(const std::string videoPath, const std::string modelPath);

        std::string videosPaths[3] = 
        {
            "video1.mp4",
            "video3.mp4",
            "video4.mp4"
        };

        std::string models[6] = 
        {
            "yolov5s.onnx",
            "yolov5m.onnx",
            "yolov5s_int8.engine",
            "yolov5m_int8.engine",
            "yolov5s_fp16.engine",
            "yolov5m_fp16.engine"
        };
};