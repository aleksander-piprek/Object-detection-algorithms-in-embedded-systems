#include "FpsCounter.hpp"

FpsCounter::FpsCounter() 
    : prevTime(std::chrono::high_resolution_clock::now()) 
{

}

double FpsCounter::update() 
{
    auto curr_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = curr_time - prevTime;
    prevTime = curr_time;
    double fps = 1.0 / diff.count();
    if (fps < 1000) 
        fpsValues.push_back(fps);

    return getAverage();
}

double FpsCounter::getAverage() const 
{
    if (fpsValues.empty()) 
        return 0.0;

    return std::accumulate(fpsValues.begin(), fpsValues.end(), 0.0) / fpsValues.size();
}