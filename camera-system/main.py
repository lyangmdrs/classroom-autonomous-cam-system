import tkinter as tk
import cv2
from multiprocessing import Process, Queue
import PIL.Image, PIL.ImageTk
import time

class GuiApplication:

    _WINDOW_NAME = "Classroom Autonomus Camera System"
    _WIDTH = 800
    _HEIGHT = 600
    _WINDOW_GEOMETRY = str(_WIDTH) + "x" + str(_HEIGHT)
    _WINDOW_UPDATE_DELAY = 15

    _MAIN_FRAME_RESIZE_FACTOR = 0.25
    _AUX_FRAME_RESIZE_FACTOR = 0.1

    def __init__(self, camera_index=0):
        self.window = tk.Tk()
        self.window.title(self._WINDOW_NAME)
        self.camera_index = camera_index

        # open video source (by default this will try to open the computer webcam)
        self.camera = FrameAcquisition(self.camera_index)
        self.camera.open_camera()

        # Create a canvas that can fit the above video source size
        self.main_canvas = tk.Canvas(self.window, width=self.camera._FRAME_WIDTH * 0.5, height=self.camera._FRAME_HEIGHT * 0.5)
        self.main_canvas.grid(row=0, column=1)

        self.face_canvas = tk.Canvas(self.window, width=self.camera._FRAME_WIDTH * 0.2, height=self.camera._FRAME_HEIGHT * 0.2)
        self.face_canvas.grid(row=0, column=0)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.camera.get_frame()

        if ret:
            main_canva_frame = cv2.resize(frame, (int(self.camera._FRAME_WIDTH * 0.5), int(self.camera._FRAME_HEIGHT * 0.5)))
            self.main_canva_frame = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(main_canva_frame))
            self.main_canvas.create_image(0, 0, image = self.main_canva_frame, anchor = tk.NW)
            
            face_canva_frame = cv2.resize(frame, (int(self.camera._FRAME_WIDTH * 0.2), int(self.camera._FRAME_HEIGHT * 0.2)))
            self.face_canva_frame = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(face_canva_frame))
            self.face_canvas.create_image(0, 0, image = self.face_canva_frame, anchor = tk.NW)

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
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
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

GuiApplication()