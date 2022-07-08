import multiprocessing
from multiprocessing import Process, Queue
import PIL.Image, PIL.ImageTk
import mediapipe as mp
import tkinter as tk
import numpy as np
import time
import cv2


class GuiApplication:

    _WINDOW_NAME = "Classroom Autonomus Camera System"
    _WIDTH = 800
    _HEIGHT = 600
    _WINDOW_GEOMETRY = str(_WIDTH) + "x" + str(_HEIGHT)
    _WINDOW_UPDATE_DELAY = 5

    _MAIN_FRAME_RESIZE_FACTOR = 0.25
    _AUX_FRAME_RESIZE_FACTOR = 0.1

    frame_width = 1280
    frame_height = 720
    frame_depth = 3
    
    raw_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)
    head_pose_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)
    hand_pose_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)

    def __init__(self, raw_frame_queue, head_pose_queue, hand_pose_queue):
        self.window = tk.Tk()
        self.window.title(self._WINDOW_NAME)

        self.raw_frame_queue = raw_frame_queue
        self.head_pose_queue = head_pose_queue
        self.hand_pose_queue = hand_pose_queue

        self.main_viewer_frame = tk.Frame(self.window, width=self.frame_width * 0.6, height=self.frame_height * 0.6)
        self.main_viewer_frame.pack(side=tk.RIGHT)

        self.debug_viewer_frame = tk.Frame(self.window, height=self.frame_height * 0.4)
        self.debug_viewer_frame.pack(side=tk.LEFT)
        
        self.main_canvas = tk.Canvas(self.main_viewer_frame, width=self.frame_width * 0.5, height=self.frame_height * 0.5)
        self.main_canvas.pack(side=tk.TOP)

        self.main_stream_label = tk.Label(self.main_viewer_frame, text="Input")
        self.main_stream_label.pack(side=tk.BOTTOM)

        self.face_canvas = tk.Canvas(self.debug_viewer_frame, width=self.frame_width * 0.2, height=self.frame_height * 0.2)
        self.face_canvas.pack(side=tk.TOP)

        self.main_stream_label = tk.Label(self.debug_viewer_frame, text="Head Pose Estimation")
        self.main_stream_label.pack(side=tk.TOP)

        self.hand_canvas = tk.Canvas(self.debug_viewer_frame, width=self.frame_width * 0.2, height=self.frame_height * 0.2)
        self.hand_canvas.pack(side=tk.TOP)

        self.main_stream_label = tk.Label(self.debug_viewer_frame, text="Hand Gesture Recognition")
        self.main_stream_label.pack(side=tk.TOP)

        self.delay = 1
        self.update()

        self.window.mainloop()

    def update(self):

        try:
            self.raw_frame = self.raw_frame_queue.get_nowait()
        except:
            pass
        try:
            self.head_pose_frame = self.head_pose_queue.get_nowait()
        except:
            pass
        try:
            self.hand_pose_frame = self.hand_pose_queue.get_nowait()
        except:
            pass

        main_canva_frame = cv2.resize(self.raw_frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5)),  interpolation = cv2.INTER_AREA)
        self.main_canva_frame = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(main_canva_frame))
        self.main_canvas.create_image(0, 0, image = self.main_canva_frame, anchor = tk.NW)
        
        face_canva_frame = cv2.resize(self.head_pose_frame, (int(self.frame_width * 0.2), int(self.frame_height * 0.2)),  interpolation = cv2.INTER_AREA)
        self.face_canva_frame = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(face_canva_frame))
        self.face_canvas.create_image(0, 0, image = self.face_canva_frame, anchor = tk.NW)
        
        hand_canva_frame = cv2.resize(self.hand_pose_frame, (int(self.frame_width * 0.2), int(self.frame_height * 0.2)),  interpolation = cv2.INTER_AREA)
        self.hand_canva_frame = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(hand_canva_frame))
        self.hand_canvas.create_image(0, 0, image = self.hand_canva_frame, anchor = tk.NW)

        self.window.after(self.delay, self.update)


class FrameAcquisition:

    _FRAME_WIDTH = 1280
    _FRAME_HEIGHT = 720

    def __init__(self, camera_index=0):
        
        self.camera = cv2.VideoCapture()
        self.camera_index = camera_index

    def open_camera(self):
        self.camera.open(self.camera_index)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._FRAME_HEIGHT)
        
        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()


class FrameProcessing:

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    def __init__(self):
        pass

    def head_pose_estimation(self, queue_input, queue_output):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                input_frame = queue_input.get()
                # input_frame.flags.writeable = False
                # frame = cv2.resize(frame, (320, 240), interpolation = cv2.INTER_AREA)
                results = holistic.process(input_frame)
                # input_frame.flags.writeable = True
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                img_h, img_w, img_c = input_frame.shape
                face_3d = []
                face_2d = []

                if results.face_landmarks:
                    # print(len(results.face_landmarks.landmark))
                    for idx, lm in enumerate(results.face_landmarks.landmark):
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
                    # if y < -10:
                    #     text = "Looking Left"
                    # elif y > 10:
                    #     text = "Looking Right"
                    # elif x < -10:
                    #     text = "Looking Down"
                    # elif x > 10:
                    #     text = "Looking Up"
                    # else:
                    #     text = "Forward"

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                    
                    cv2.line(input_frame, p1, p2, (255, 0, 0), 3)

                self.mp_drawing.draw_landmarks(input_frame,
                                               results.face_landmarks,
                                               self.mp_holistic.FACEMESH_TESSELATION,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mp_drawing_styles
                                               .get_default_face_mesh_tesselation_style())
                queue_output.put(input_frame)


    def hand_gesture_recognition(self, queue_input, queue_output):
        while True:
            input_frame = queue_input.get()
            output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
            queue_output.put(output_frame)
            

class FrameServer:
    
    def __init__(self, queue_raw_frame_server_input, queue_raw_frame_server_output, 
                queue_hand_gesture_recognition_input, queue_hand_gesture_recognition_output, 
                queue_head_pose_estimation_input, queue_head_pose_estimation_output,
                frame_step=5):

        self.frame_step = frame_step
        self.queue_raw_frame_server_input = queue_raw_frame_server_input
        self.queue_raw_frame_server_output = queue_raw_frame_server_output
        self.queue_hand_gesture_recognition_input = queue_hand_gesture_recognition_input
        self.queue_hand_gesture_recognition_output = queue_hand_gesture_recognition_output
        self.queue_head_pose_estimation_input = queue_head_pose_estimation_input
        self.queue_head_pose_estimation_output = queue_head_pose_estimation_output
    
    def start_server(self):
        frame_counter = 0
        while True:
            frame = self.queue_raw_frame_server_input.get()
            try:
                self.queue_raw_frame_server_output.put_nowait(frame)
                frame_counter = (frame_counter + 1) % self.frame_step
                if frame_counter == 0:
                    self.queue_head_pose_estimation_input.put_nowait(frame)
                    self.queue_hand_gesture_recognition_input.put_nowait(frame)
            except:
                continue


class ProcessManager:

    _ALL_QUEUES = []
    
    def __init__(self, queues_size):
        self.queue_raw_frame_server_input = Queue(queues_size)
        self.queue_raw_frame_server_output = Queue(queues_size)

        self.queue_hand_gesture_recognition_input = Queue(queues_size)
        self.queue_hand_gesture_recognition_output = Queue(queues_size)

        self.queue_head_pose_estimation_input = Queue(queues_size)
        self.queue_head_pose_estimation_output = Queue(queues_size)

        self._ALL_QUEUES = [self.queue_raw_frame_server_input,
                            self.queue_raw_frame_server_output,
                            self.queue_hand_gesture_recognition_input,
                            self.queue_hand_gesture_recognition_output,
                            self.queue_head_pose_estimation_input,
                            self.queue_head_pose_estimation_output,]


    def close_all_queues(self):
        for q in self._ALL_QUEUES:
            q.close()

def acquirer(queue):
    camera = FrameAcquisition()
    camera.open_camera()
    frame = np.zeros((camera._FRAME_WIDTH, camera._FRAME_HEIGHT, 3), np.uint8)
    while True:
        ret, frame = camera.get_frame()
        try:
            queue.put_nowait(frame)
        except:
            continue

def frame_server(input_queue, output_queue, head_pose_queue, hand_gesture_queue):
    frame_stepping = 5
    frame_counter = 0
    while True:
        frame = input_queue.get()
        try:
            output_queue.put_nowait(frame)
            frame_counter = (frame_counter + 1) % frame_stepping
            if frame_counter == 0:
                head_pose_queue.put_nowait(frame)
                hand_gesture_queue.put_nowait(frame)
        except:
            continue

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    pm = ProcessManager(5)
    fp = FrameProcessing()
    frame_server = FrameServer(pm.queue_raw_frame_server_input, pm.queue_raw_frame_server_output,
                               pm.queue_hand_gesture_recognition_input, pm.queue_hand_gesture_recognition_output,
                               pm.queue_head_pose_estimation_input, pm.queue_head_pose_estimation_output)

    acquirer_process = Process(target=acquirer, args=(pm.queue_raw_frame_server_input,))
    acquirer_process.start()

    server_process = Process(target=frame_server, args=(pm.queue_raw_frame_server_input, pm.queue_raw_frame_server_output, 
                                                        pm.queue_head_pose_estimation_input, pm.queue_hand_gesture_recognition_input,))
    server_process.start()

    head_estimation_process = Process(target=fp.head_pose_estimation, args=(pm.queue_head_pose_estimation_input, 
                                                                            pm.queue_head_pose_estimation_output,))
    head_estimation_process.start()

    hand_recognition_process = Process(target=fp.hand_gesture_recognition, args=(pm.queue_hand_gesture_recognition_input,
                                                                                 pm.queue_hand_gesture_recognition_output))
    hand_recognition_process.start()

    gui = GuiApplication(pm.queue_raw_frame_server_output, pm.queue_head_pose_estimation_output, pm.queue_hand_gesture_recognition_output)

    pm.close_all_queues()

    acquirer_process.terminate()
    server_process.terminate()
    head_estimation_process.terminate()
    hand_recognition_process.terminate()