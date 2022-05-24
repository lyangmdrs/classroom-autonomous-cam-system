#ifndef PAN_TILT_DRIVER_H
#define PAN_TILT_DRIVER_H

constexpr float SERVO_PERIOD = 0.02;

constexpr int PAN_MOTOR_PIN = 9;
constexpr int TILT_MOTOR_PIN = 10;

constexpr int PAN_LIMIT_MIN = 600;
constexpr int PAN_MIDDLE = 1450;
constexpr int PAN_LIMIT_MAX = 2300;

constexpr int SAFE_PAN_TON(int t)
{
    return (t < PAN_LIMIT_MIN)? PAN_LIMIT_MIN : (t > PAN_LIMIT_MAX)? PAN_LIMIT_MAX : t;
} 

constexpr int TILT_LIMIT_MIN = 700;
constexpr int TILT_MIDDLE = 1500;
constexpr int TILT_LIMIT_MAX = 1800;

constexpr int SAFE_TILT_TON(int t)
{
    return (t < TILT_LIMIT_MIN)? TILT_LIMIT_MIN : (t > TILT_LIMIT_MAX)? TILT_LIMIT_MAX : t;
}

constexpr int CONVERT_SECOND_TO_MILLISECOND(float s)
{
    return int(s * 1000);
}

class PanTiltDriver
{
private:
    int panTimeOn;
    int tiltTimeOn;

    void configTimersAndRegisters();
    void interruptionHandler();
public:
    PanTiltDriver();
    void begin();
    void setPanTimeOn(int value);
    void setTiltTimeOn(int value);
    void setInitialPanPosition();
    void setInitialTiltPosition();
    void incrementPanTimeOn(int step);
    void incrementTiltTimeOn(int step);
    int getPanTimeOn();
    int getTiltTimeOn();
};

#endif