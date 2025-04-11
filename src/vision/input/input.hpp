#pragma once

#ifndef VISION_INPUT_HPP
#define VISION_INPUT_HPP

#include <string>
#include <opencv2/opencv.hpp>

namespace Vision
{
    class Input
    {
        public:
            virtual ~Input() = default;
    };
}

#endif // VISION_INPUT_HPP