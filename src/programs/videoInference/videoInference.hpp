#include "string"

class VideoInference
{
    public:
        VideoInference() = default;
        VideoInference(const std::string& configPath);
        ~VideoInference() = default;

        void run();

    private:
        bool loadVideoInferenceConfig(const std::string& configPath);
        std::string modelPath;
        std::string videoPath;
        std::string classNamesPath;
        float confThreshold;
        float nmsThreshold;
};