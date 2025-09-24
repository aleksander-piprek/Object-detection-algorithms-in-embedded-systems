#pragma once

#ifndef WINDOW_OUTPUT_HPP
#define WINDOW_OUTPUT_HPP

#include "src/components/output/baseOutput.hpp"

class WindowOutput : public BaseOutput
{
    public:
        WindowOutput(const std::string& windowName);
        ~WindowOutput() = default;

        void write(const cv::Mat& frame) override;
        void write(const cv::Mat& frame, const double& avgFps) override;
        void close() override;
        
    private:
        std::string windowName;
        int frameCount;
        std::chrono::high_resolution_clock::time_point startTime;
};

#endif // WINDOW_OUTPUT_HPP