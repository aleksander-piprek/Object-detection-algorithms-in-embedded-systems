#include "gtest/gtest.h"

#include "TestImageInput/TestImageInput.cpp"
#include "TestVideoInput/TestVideoInput.cpp"
#include "TestWindowOutput/TestWindowOutput.cpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}