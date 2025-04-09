#include "gtest/gtest.h"

#include "TestImage.hpp"

TEST(TestImage, testLoadImage)
{
    TestImage testImage("../../resources/images/Crow.JPG");
    EXPECT_EQ(testImage.getWidth(), 100);
    EXPECT_EQ(testImage.getHeight(), 100);
}