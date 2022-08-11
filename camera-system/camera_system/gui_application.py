"""Module that creates a graphic user interface."""
import tkinter as tk
import queue
import numpy as np
import cv2
import PIL.Image
import PIL.ImageTk

class GuiApplication:
    """Class that creates a graphic user interface."""

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
    hand_gesture_name = ""

    raw_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)
    head_pose_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)
    hand_pose_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)
    processed_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)

    def __init__(self, raw_frame_queue, head_pose_queue,
                 hand_pose_queue, pipe_connection, output_frame_queue):
        self.window = tk.Tk()
        self.window.title(self._WINDOW_NAME)

        self.raw_frame_queue = raw_frame_queue
        self.head_pose_queue = head_pose_queue
        self.hand_pose_queue = hand_pose_queue
        self.pipe_connection = pipe_connection
        self.processed_queue = output_frame_queue

        self.main_viewer_frame = tk.Frame(self.window)
        self.main_viewer_frame.pack(side=tk.TOP)

        self.debug_viewer_frame = tk.Frame(self.window)
        self.debug_viewer_frame.pack(side=tk.BOTTOM)

        self.head_pose_viewer_frame = tk.Frame(self.debug_viewer_frame)
        self.head_pose_viewer_frame.pack(side=tk.LEFT)

        self.hand_gesture_viwer_frame = tk.Frame(self.debug_viewer_frame)
        self.hand_gesture_viwer_frame.pack(side=tk.RIGHT)

        self.input_viewer_frame = tk.Frame(self.main_viewer_frame)
        self.input_viewer_frame.pack(side=tk.LEFT)

        self.output_viewer_frame = tk.Frame(self.main_viewer_frame)
        self.output_viewer_frame.pack(side=tk.RIGHT)

        self.input_cava = tk.Canvas(self.input_viewer_frame, width=self.frame_width * 0.4,
                                     height=self.frame_height * 0.4)
        self.input_cava.pack(side=tk.TOP)

        self.input_label = tk.Label(self.input_viewer_frame, text="Input")
        self.input_label.pack(side=tk.BOTTOM)

        self.output_canva = tk.Canvas(self.output_viewer_frame, width=self.frame_width * 0.4,
                                     height=self.frame_height * 0.4)
        self.output_canva.pack(side=tk.TOP)

        self.output_label = tk.Label(self.output_viewer_frame, text="Output")
        self.output_label.pack(side=tk.BOTTOM)

        self.head_canvas = tk.Canvas(self.head_pose_viewer_frame,
                                     width=self.frame_width * 0.2,
                                     height=self.frame_height * 0.2)
        self.head_canvas.pack(side=tk.TOP)

        self.head_pose_label = tk.Label(self.head_pose_viewer_frame, text="Head Pose Estimation")
        self.head_pose_label.pack(side=tk.BOTTOM)

        self.hand_canvas = tk.Canvas(self.hand_gesture_viwer_frame,
                                     width=self.frame_width * 0.2,
                                     height=self.frame_height * 0.2)
        self.hand_canvas.pack(side=tk.TOP)

        self.gesture_name_label = tk.Label(self.hand_gesture_viwer_frame,
                                           font="Courier 18 bold", text="",
                                           bg="black", fg="green")
        self.gesture_name_label.pack(side=tk.TOP, fill=tk.BOTH)

        self.hand_gesture_label = tk.Label(self.hand_gesture_viwer_frame,
                                           text="Hand Gesture Recognition")
        self.hand_gesture_label.pack(side=tk.TOP)

        self.delay = 1
        self.update()

        self.window.mainloop()

    def update(self):
        """Responsible for updating the graphical interface elements."""
        try:
            self.raw_frame = self.raw_frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.head_pose_frame = self.head_pose_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.hand_pose_frame = self.hand_pose_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.processed_frame = self.processed_queue.get_nowait()
        except queue.Empty:
            pass

        if self.pipe_connection.poll():
            self.hand_gesture_name = self.pipe_connection.recv()
            self.gesture_name_label.configure(text=self.hand_gesture_name)


        input_canva_frame = cv2.resize(self.raw_frame,
                                      (int(self.frame_width * 0.4),
                                      int(self.frame_height * 0.4)),
                                      interpolation = cv2.INTER_AREA)

        self.input_canva_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(input_canva_frame))
        self.input_cava.create_image(0, 0, image=self.input_canva_frame, anchor = tk.NW)

        heap_pose_frame = PIL.Image.fromarray(self.head_pose_frame)
        self.face_canva_frame = PIL.ImageTk.PhotoImage(image=heap_pose_frame)
        self.head_canvas.create_image(0, 0, image=self.face_canva_frame, anchor = tk.NW)

        hand_canva_frame = cv2.resize(self.hand_pose_frame,
                                      (int(self.frame_width * 0.2),
                                      int(self.frame_height * 0.2)),
                                      interpolation = cv2.INTER_AREA)
        self.hand_canva_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(hand_canva_frame))
        self.hand_canvas.create_image(0, 0, image=self.hand_canva_frame, anchor = tk.NW)

        output_canva_frame = cv2.resize(self.processed_frame,
                                        (int(self.frame_width * 0.4),
                                        int(self.frame_height * 0.4)),
                                        interpolation = cv2.INTER_AREA)
        self.output_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(output_canva_frame))
        self.output_canva.create_image(0, 0, image=self.output_frame, anchor = tk.NW)


        self.window.after(self.delay, self.update)
