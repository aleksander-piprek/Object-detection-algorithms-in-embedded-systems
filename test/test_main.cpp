#include "gtest/gtest.h"

#include "TestWindow/TestWindow.cpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}