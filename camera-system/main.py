from ctypes import sizeof
import multiprocessing
from multiprocessing import Process, Queue
import PIL.Image, PIL.ImageTk
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

        self.main_canvas = tk.Canvas(self.window, width=self.frame_width * 0.5, height=self.frame_height * 0.5)
        self.main_canvas.grid(row=0, column=1)

        self.face_canvas = tk.Canvas(self.window, width=self.frame_width * 0.2, height=self.frame_height * 0.2)
        self.face_canvas.grid(row=0, column=0)

        self.hand_canvas = tk.Canvas(self.window, width=self.frame_width * 0.2, height=self.frame_height * 0.2)
        self.hand_canvas.grid(row=1, column=0)

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

    def __init__(self):
        pass

    def head_pose_estimation(self, queue_input, queue_output):
        while True:
            input_frame = queue_input.get()
            output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
            queue_output.put(output_frame)

    def hand_gesture_recognition(self, queue_input, queue_output):
        while True:
            input_frame = queue_input.get()
            output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
            queue_output.put(output_frame)
            


class FrameServer:

    def __init__(self) -> None:
        pass


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
            print("erro")
            continue

def frame_server(input_queue, output_queue, head_pose_queue, hand_gesture_queue):
    while True:
        frame = input_queue.get()
        try:
            output_queue.put_nowait(frame)
            head_pose_queue.put_nowait(frame)
            hand_gesture_queue.put_nowait(frame)
        except:
            continue

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    pm = ProcessManager(5)
    fp = FrameProcessing()

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
    print("fecha queue")

    pm.close_all_queues()

    acquirer_process.terminate()
    server_process.terminate()
    head_estimation_process.terminate()
    hand_recognition_process.terminate()