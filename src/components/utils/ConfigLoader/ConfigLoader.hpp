#include <string>
#include <unordered_map>

class ConfigLoader 
{
    public:
        ConfigLoader();

        static std::unordered_map<std::string, std::string> loadConfig(const std::string& configPath);
    
    private:
        static std::string getModelPath(const std::string& model_name);
        static std::string getVideoPath(const std::string& video_name);        
        static std::string getClassNamesPath(const std::string& class_name);
};