import cv2
import mediapipe as mp
import numpy as np
import time
import SerialCommunication as sm


CAMERA_INDEX = 0
HD_WIDTH = 1280
HD_HEIGHT = 720
DRAW_RULES = True

videoInput = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
videoInput.set(cv2.CAP_PROP_FRAME_WIDTH, HD_WIDTH)
videoInput.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_HEIGHT)
mp_drawing = mp.solutions.drawing_utils
#mp_holistic = mp.solutions.holistic
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def main():
    support_driver = sm.get_driver()

    # Media Pipe Settings
    #holistic = mp_holistic.Holistic(min_detection_confidence=0.55, min_tracking_confidence=0.55)
        
    while videoInput.isOpened():
        
        success, image = videoInput.read()

        start = time.time()

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        results = face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
            

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #mp_drawing.draw_landmarks(
            #            image=image,
            #            landmark_list=face_landmarks,
            #            connections=mp_face_mesh.FACEMESH_CONTOURS,
            #            landmark_drawing_spec=drawing_spec,
            #            connection_drawing_spec=drawing_spec)

            
            #if (results.pose_landmarks):
            #nose_landmark_x = int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * frame_width)
            #nose_landmark_y = int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * frame_height)

            #x_axis_distance = frame_width//2 - nose_landmark_x
            #y_axis_distance = nose_landmark_y - frame_height//2

            x_axis_distance = p1[0] - img_w//2
            y_axis_distance = p1[1] - img_h//2

            sm.send_command(support_driver, x_axis_distance//10, y_axis_distance//4)
            #sm.send_command(support_driver, x_axis_distance//10, 0)
            
            if (DRAW_RULES is True):
                print("Distance: ({},{})".format(x_axis_distance, y_axis_distance))
                cv2.line(image, (img_w//2, 0), (img_w//2, img_h), (255, 0, 255), 1)
                cv2.line(image, (0, img_h//2), (img_w, img_h//2), (255, 0, 255), 1)
                #cv2.line(image, (nose_landmark_x, nose_landmark_y), (nose_landmark_x, img_h//2), (0, 255, 0), 1)
                #cv2.line(image, (nose_landmark_x, nose_landmark_y), (img_w//2, nose_landmark_y), (0, 255, 0), 1)
                #cv2.circle(image, (nose_landmark_x, nose_landmark_y), 2, (255, 0, 0), -1)
                cv2.circle(image, (img_w//2, img_h//2), 3, (0, 0, 255), -1)
            
            
            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            #print("FPS: ", fps)
            #print('Nose Points: ', p1)

            cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                
        cv2.imshow("Viewer - Press 'ESC' to close", image)

        if (cv2.waitKey(5) & 0xFF == 27): # Press ESC
            
            videoInput.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()