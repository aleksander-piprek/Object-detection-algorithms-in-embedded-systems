#pragma once
#include <vector>
#include <chrono>
#include <numeric>

class FpsCounter 
{
    public:
        FpsCounter();
        double update();
        double getAverage() const;

    private:
        std::vector<double> fpsValues;
        std::chrono::high_resolution_clock::time_point prevTime;
        
};