from unittest import result
import cv2
import mediapipe as mp
import numpy as np
import time
import SerialCommunication as sm


CAMERA_INDEX = 0
HD_WIDTH = 1280
HD_HEIGHT = 720
DRAW_RULES = True

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, HD_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_HEIGHT)

def detect_face():
    pass


def main():
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.95) as face_detection:
        while cap.isOpened():
        
            start = time.time()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
        
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)
        
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    main()