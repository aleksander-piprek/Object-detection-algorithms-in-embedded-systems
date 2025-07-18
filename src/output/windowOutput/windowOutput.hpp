#pragma once

#ifndef WINDOW_OUTPUT_HPP
#define WINDOW_OUTPUT_HPP

#include "src/output/baseOutput.hpp"

class WindowOutput : public BaseOutput
{
    public:
        WindowOutput(const std::string& windowName);
        ~WindowOutput() = default;

        void write(const cv::Mat& frame) override;
        void close() override;
        
    private:
        std::string windowName;
};

#endif // WINDOW_OUTPUT_HPP