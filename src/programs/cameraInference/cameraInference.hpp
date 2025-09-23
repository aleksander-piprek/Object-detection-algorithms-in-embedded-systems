#include "string"

class CameraInference
{
    public:
        CameraInference() = default;
        CameraInference(const std::string& configPath);
        ~CameraInference() = default;

        void run();

    private:
        bool loadCameraInferenceConfig(const std::string& configPath);
        std::string modelPath;
        std::string cameraPath;
        std::string classNamesPath;
        float confThreshold;
        float nmsThreshold;
};