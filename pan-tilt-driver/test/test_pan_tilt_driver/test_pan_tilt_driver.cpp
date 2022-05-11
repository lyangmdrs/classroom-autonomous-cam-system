#include <Arduino.h>
#include <PanTiltDriver.h>
#include <unity.h>

void test_support_constructor(void)
{
    PanTiltDriver cameraSupport;
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_MIDDLE);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_MIDDLE);
}

void test_support_setting_values(void)
{
    PanTiltDriver cameraSupport;
    cameraSupport.setPanTimeOn(PAN_LIMIT_MIN);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_LIMIT_MIN);
    cameraSupport.setPanTimeOn(PAN_LIMIT_MAX);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_LIMIT_MAX);
    cameraSupport.setTiltTimeOn(TILT_LIMIT_MIN);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_LIMIT_MIN);
    cameraSupport.setTiltTimeOn(TILT_LIMIT_MAX);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_LIMIT_MAX);
}

void test_support_increment_values(void)
{
    PanTiltDriver cameraSupport;
    cameraSupport.incrementPanTimeOn(10);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_MIDDLE + 10);
    cameraSupport.incrementPanTimeOn(-10);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_MIDDLE);
    cameraSupport.incrementTiltTimeOn(10);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_MIDDLE + 10);
    cameraSupport.incrementTiltTimeOn(-10);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_MIDDLE);
}

void test_safe_values(void)
{
    PanTiltDriver cameraSupport;
    cameraSupport.setPanTimeOn(PAN_LIMIT_MAX + 1);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_LIMIT_MAX);
    cameraSupport.incrementPanTimeOn(10);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_LIMIT_MAX);
    cameraSupport.setTiltTimeOn(TILT_LIMIT_MAX + 1);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_LIMIT_MAX);
    cameraSupport.incrementTiltTimeOn(10);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_LIMIT_MAX);
    cameraSupport.setPanTimeOn(PAN_LIMIT_MIN - 1);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_LIMIT_MIN);
    cameraSupport.incrementPanTimeOn(-10);
    TEST_ASSERT_EQUAL(cameraSupport.getPanTimeOn(), PAN_LIMIT_MIN);
    cameraSupport.setTiltTimeOn(TILT_LIMIT_MIN - 1);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_LIMIT_MIN);
    cameraSupport.incrementTiltTimeOn(-10);
    TEST_ASSERT_EQUAL(cameraSupport.getTiltTimeOn(), TILT_LIMIT_MIN);
}

void setup()
{
    UNITY_BEGIN();
    RUN_TEST(test_support_constructor);
    RUN_TEST(test_support_setting_values);
    RUN_TEST(test_support_increment_values);
    RUN_TEST(test_safe_values);
    UNITY_END();
}

void loop() {}