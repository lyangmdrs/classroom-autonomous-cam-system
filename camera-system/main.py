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

def detect_head_pose():
    pass


def detect_face(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.95) as face_detection:
        
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []



            if results.detections:
                for detection in results.detections:
                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    nose = detection.location_data.relative_keypoints[2]
                    mouth = detection.location_data.relative_keypoints[3]
                    left_ear = detection.location_data.relative_keypoints[4]
                    right_ear = detection.location_data.relative_keypoints[5]
                    
                    cv2.circle(image,(int(left_eye.x * img_w), int(left_eye.y * img_h)), 5, (0,255,0), -1)
                    cv2.circle(image,(int(right_eye.x * img_w), int(right_eye.y * img_h)), 5, (0,255,0), -1)
                    cv2.circle(image,(int(nose.x * img_w), int(nose.y * img_h)), 5, (0,255,0), -1)
                    cv2.circle(image,(int(mouth.x * img_w), int(mouth.y * img_h)), 5, (0,255,0), -1)
                    cv2.circle(image,(int(left_ear.x * img_w), int(left_ear.y * img_h)), 5, (0,255,0), -1)
                    cv2.circle(image,(int(right_ear.x * img_w), int(right_ear.y * img_h)), 5, (0,255,0), -1)
                    
                    # mp_drawing.draw_detection(image, detection)
    
    return image
            


def main():
    while cap.isOpened():
        
            start = time.time()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            image =  cv2.flip(image, 1)
            cv2.putText(image, f'FPS: {int(fps)}', (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)

            image = detect_face(image)

            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    main()