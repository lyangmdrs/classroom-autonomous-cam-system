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
    _WINDOW_UPDATE_DELAY = 15

    _MAIN_FRAME_RESIZE_FACTOR = 0.25
    _AUX_FRAME_RESIZE_FACTOR = 0.1

    frame_width = 1280
    frame_height = 720
    frame_depth = 3
    frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)

    def __init__(self, queue):
        self.window = tk.Tk()
        self.window.title(self._WINDOW_NAME)
        self.main_queue = queue
        self.main_canvas = tk.Canvas(self.window, width=self.frame_width * 0.5, height=self.frame_height * 0.5)
        self.main_canvas.grid(row=0, column=1)

        self.face_canvas = tk.Canvas(self.window, width=self.frame_width * 0.2, height=self.frame_height * 0.2)
        self.face_canvas.grid(row=0, column=0)

        self.hand_canvas = tk.Canvas(self.window, width=self.frame_width * 0.2, height=self.frame_height * 0.2)
        self.hand_canvas.grid(row=1, column=0)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):

        self.frame = self.main_queue.get()

        main_canva_frame = cv2.resize(self.frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5)),  interpolation = cv2.INTER_AREA)
        self.main_canva_frame = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(main_canva_frame))
        self.main_canvas.create_image(0, 0, image = self.main_canva_frame, anchor = tk.NW)
        
        face_canva_frame = cv2.resize(self.frame, (int(self.frame_width * 0.2), int(self.frame_height * 0.2)),  interpolation = cv2.INTER_AREA)
        self.face_canva_frame = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(face_canva_frame))
        self.face_canvas.create_image(0, 0, image = self.face_canva_frame, anchor = tk.NW)
        
        hand_canva_frame = cv2.resize(self.frame, (int(self.frame_width * 0.2), int(self.frame_height * 0.2)),  interpolation = cv2.INTER_AREA)
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

    def __init__(self) -> None:
        pass


class FrameServer:

    def __init__(self) -> None:
        pass


class ProcessManager:

    queue_frame_server_input = Queue()
    queue_hand_gesture_recognition_input = Queue()
    queue_head_pose_estimation_input = Queue()


    def __init__(self) -> None:
        pass

def acquirer(queue):
    camera = FrameAcquisition()
    camera.open_camera()
    frame = np.zeros((camera._FRAME_WIDTH, camera._FRAME_HEIGHT, 3), np.uint8)
    while True:
        ret, frame = camera.get_frame()
        queue.put_nowait(frame)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    frame_queue = Queue()
    acquirer_process = Process(target=acquirer, args=(frame_queue,))
    acquirer_process.start()

    GuiApplication(frame_queue)
    print("fecha queue")
    frame_queue.close()
    acquirer_process.terminate()