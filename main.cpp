#include "src/vision/window/window.hpp"
#include "src/vision/input/image/image.hpp"

int main()
{
    Vision::Image image("../resources/images/Wagtail.JPG");
    image.showImage();
    return 0;
}