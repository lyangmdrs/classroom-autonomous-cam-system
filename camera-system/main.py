import cv2
import mediapipe as mp
import numpy as np
import time
import SerialCommunication as sm


CAMERA_INDEX = 0
HD_WIDTH = 1280
HD_HEIGHT = 720
DRAW_RULES = True

videoInput = cv2.VideoCapture(CAMERA_INDEX)
videoInput.set(cv2.CAP_PROP_FRAME_WIDTH, HD_WIDTH)
videoInput.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_HEIGHT)
mp_drawing = mp.solutions.drawing_utils

def detect_face(frame):
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.95) as face_detection:
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame)
        return results.detections


def main():

    while videoInput.isOpened():
        
        start = time.time()
        
        success, image = videoInput.read()

        detections = detect_face(image)
        for detection in detections:
            mp_drawing.draw_detection(image, detection)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)

        cv2.imshow("Viewer - Press 'ESC' to close", image)
        
        if (cv2.waitKey(5) & 0xFF == 27): # Press ESC
            
            videoInput.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()