#include <Arduino.h>
#include "PanTiltDriver.h"

PanTiltDriver::PanTiltDriver()
{
    panTimeOn = PAN_MIDDLE;
    tiltTimeOn = TILT_MIDDLE;
}

void PanTiltDriver::configTimersAndRegisters()
{
  TCCR2A = 0xA3; // Sets PWM
  TCCR2B = 0x04; // Sets Prescaler 
  TIMSK2 = 0x01; // Sets Timer2 interruption
  sei(); // Enables global interruption
}

void PanTiltDriver::begin()
{
    pinMode(PAN_MOTOR_PIN, OUTPUT);
    pinMode(TILT_MOTOR_PIN, OUTPUT);
    configTimersAndRegisters();
}

void PanTiltDriver::interruptionHandler()
{
    // Not Implemented
}

void PanTiltDriver::setPanTimeOn(int value)
{
    panTimeOn = SAFE_PAN_TON(value);
}

void PanTiltDriver::setTiltTimeOn(int value)
{
    tiltTimeOn = SAFE_TILT_TON(value);
}

void PanTiltDriver::setInitialPanPosition()
{
    panTimeOn = SAFE_PAN_TON(PAN_MIDDLE);
}

void PanTiltDriver::setInitialTiltPosition()
{
    tiltTimeOn = SAFE_TILT_TON(TILT_MIDDLE);
}

void PanTiltDriver::incrementPanTimeOn(int step)
{
    panTimeOn = SAFE_PAN_TON(panTimeOn + step);
}

void PanTiltDriver::incrementTiltTimeOn(int step)
{
    tiltTimeOn = SAFE_TILT_TON(tiltTimeOn + step);
}

int PanTiltDriver::getPanTimeOn()
{
    return panTimeOn;
}

int PanTiltDriver::getTiltTimeOn()
{
    return tiltTimeOn;
}