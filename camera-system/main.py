import cv2
import mediapipe as mp
import SerialCommunication as sm
import time

CAMERA_INDEX = 0
HD_WIDTH = 1280
HD_HEIGHT = 720

videoInput = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
videoInput.set(cv2.CAP_PROP_FRAME_WIDTH, HD_WIDTH)
videoInput.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_HEIGHT)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    support_driver = sm.get_driver()
    with mp_holistic.Holistic(min_detection_confidence=0.55, min_tracking_confidence=0.55) as holistic:
        
        while videoInput.isOpened():
            
            _, raw_frame = videoInput.read()      
            frame_height, frame_width, _ = raw_frame.shape

            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)

            if (results.pose_landmarks):
                nose_landmark_x = int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * frame_width)
                nose_landmark_y = int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * frame_height)

                x_axis_distance = frame_width//2 - nose_landmark_x
                y_axis_distance = nose_landmark_y - frame_height//2

                #print("Distance: {}x{}".format(x_axis_distance, y_axis_distance))
                sm.send_command(support_driver, x_axis_distance//10, y_axis_distance//3)
                #cv2.line(raw_frame, (frame_width//2, 0), (frame_width//2, frame_height), (255, 0, 255), 1)
                #cv2.line(raw_frame, (0, frame_height//2), (frame_width, frame_height//2), (255, 0, 255), 1)
                #cv2.line(raw_frame, (nose_landmark_x, nose_landmark_y), (nose_landmark_x, frame_height//2), (0, 255, 0), 1)
                #cv2.line(raw_frame, (nose_landmark_x, nose_landmark_y), (frame_width//2, nose_landmark_y), (0, 255, 0), 1)
                #cv2.circle(raw_frame, (nose_landmark_x, nose_landmark_y), 2, (255, 0, 0), -1)
                #cv2.circle(raw_frame, (frame_width//2, frame_height//2), 3, (0, 0, 255), -1)


            cv2.imshow("Viewer - Press 'ESC' to close", cv2.flip(raw_frame, 1))

            if (cv2.waitKey(5) & 0xFF == 27): # Press ESC
                
                videoInput.release()
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()