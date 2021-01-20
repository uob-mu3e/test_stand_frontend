// calc_test.cpp
#include <gtest/gtest.h>
#include "mudaq_dummy.h"

dummy_mudaq::DummyMudaqDevice * mup;

TEST(MudaqDeviceTest, read_register_rw) {
    mup = new dummy_mudaq::DummyMudaqDevice("/dev/mudaq0");
    dummy_mudaq::DummyMudaqDevice & mu = *mup;
    mu.write_register(0x1, 0x1);
    ASSERT_EQ(0x1, mu.read_register_rw(0x1));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
