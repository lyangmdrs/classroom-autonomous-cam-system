#include <Arduino.h>
#include <PanTiltDriver.h>

#include "PanTiltUtils.h"

PanTiltDriver cameraSupport;

void configTimersAndRegisters();

void setTrajectory(int* ton,
                   int initialPosition,
                   int finalPosition,
                   float interval);

void demonstration();

bool messagePaser(String* message, int* phiValue, int* thetaValue);

void setup() 
{
  Serial.begin(SERIAL_BAUD_RATE);
  Serial.setTimeout(SERIAL_TIMEOUT);
  
  cameraSupport.begin();
  
  WAIT_FOR_SERIAL();
  
  DEBUG_MSG("Setup done!");
}

void loop() {}

// Interruption Handler
ISR(TIMER2_OVF_vect)
{
  static int counter = 0x00;
  
  counter = counter + 1;

  if (CONVERT_SECOND_TO_MILLISECOND(SERVO_PERIOD) == counter)
  {
    counter = 0;

    digitalWrite(PAN_MOTOR_PIN, HIGH);
    delayMicroseconds(cameraSupport.getPanTimeOn());
    digitalWrite(PAN_MOTOR_PIN, LOW);

    digitalWrite(TILT_MOTOR_PIN, HIGH);
    delayMicroseconds(cameraSupport.getTiltTimeOn());
    digitalWrite(TILT_MOTOR_PIN, LOW);
  }
}

void serialEvent()
{
  String received = String("");
  int iPanValue = 0;
  int iTiltValue = 0;

  if(Serial.available() > 0)
  {
    received = Serial.readStringUntil(SERIAL_TERMINATOR);
  }

  bool success = messagePaser(&received, &iPanValue, &iTiltValue);
  
  if (success)
  {
    cameraSupport.incrementPanTimeOn(iPanValue);
    cameraSupport.incrementTiltTimeOn(iTiltValue);
    Serial.print(received);
    Serial.println(SERIAL_TERMINATOR);
  }
  else
  {
    Serial.println("Invalid!");
  }
}

bool messagePaser(String* pMessage, int* pPanValue, int* pTiltValue)
{
  unsigned int uSeparatorIndex = 0;

  if (pMessage->length() == VALID_MSG_LENGTH) 
  {
    uSeparatorIndex = pMessage->indexOf(SERIAL_SEPARATOR);
    
    String sPhiValue = pMessage->substring(0, uSeparatorIndex);
    String sThetaValue = pMessage->substring(uSeparatorIndex + 1, pMessage->length());
    
    *pPanValue = sPhiValue.toInt();
    *pTiltValue = sThetaValue.toInt();
    
    return true;
  }
  else
  {
    Serial.print("Received: ");
    Serial.println(*pMessage);
    Serial.print("Length: ");
    Serial.println(pMessage->length());
  }

  return false;
}