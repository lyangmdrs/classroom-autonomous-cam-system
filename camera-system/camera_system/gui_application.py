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
    _WIDTH = 1040
    _HEIGHT = 510
    _WINDOW_GEOMETRY = str(_WIDTH) + "x" + str(_HEIGHT)
    _WINDOW_UPDATE_DELAY = 5

    _MAIN_FRAME_RESIZE_FACTOR = 0.25
    _AUX_FRAME_RESIZE_FACTOR = 0.1

    frame_width = 1280
    frame_height = 720
    frame_depth = 3
    hand_gesture_name = ""

    initial_frame = np.reshape(np.repeat(155, frame_height*frame_width*frame_depth),
                               (frame_height, frame_width, frame_depth))


    raw_frame = np.asarray(initial_frame, np.uint8)
    raw_frame = cv2.putText(raw_frame, "STARTING...", (150, 450), 0, 6,  (8,77,110), 6, 16, False)
    head_pose_frame = cv2.resize(raw_frame, (frame_width // 5, frame_height // 5), interpolation=3)
    hand_pose_frame = raw_frame
    processed_frame = raw_frame

    def __init__(self, raw_frame_queue, head_pose_queue,
                 hand_pose_queue, pipe_connection, output_frame_queue):
        self.window = tk.Tk()
        self.window.title(self._WINDOW_NAME)
        self.window.geometry(self._WINDOW_GEOMETRY)

        self.raw_frame_queue = raw_frame_queue
        self.head_pose_queue = head_pose_queue
        self.hand_pose_queue = hand_pose_queue
        self.pipe_connection = pipe_connection
        self.processed_queue = output_frame_queue

        self.main_viewer_frame = tk.Frame(self.window, height=self._HEIGHT//2)
        self.main_viewer_frame.pack(side=tk.TOP, ipady=10, fill=tk.X)

        self.debug_viewer_frame = tk.Frame(self.window, height=self._HEIGHT//2)
        self.debug_viewer_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.head_pose_viewer_frame = tk.Frame(self.debug_viewer_frame, 
                                               width=self.frame_width * 0.2 + 10)
        self.head_pose_viewer_frame.pack(side=tk.LEFT, ipadx=5,
                                         fill=tk.Y, expand=True)

        self.hand_gesture_viwer_frame = tk.Frame(self.debug_viewer_frame,
                                                 width=self.frame_width * 0.2 + 10)
        self.hand_gesture_viwer_frame.pack(side=tk.LEFT, ipadx=5,
                                           fill=tk.Y, expand=True)

        self.controls_and_info_frame = tk.Frame(self.debug_viewer_frame, width=self._WIDTH//2 - 5)
        self.controls_and_info_frame.pack(side=tk.LEFT, anchor=tk.W, fill=tk.BOTH, expand=True)

        self.command_frame = tk.Frame(self.controls_and_info_frame, width=self._WIDTH//2 - 10)
        self.command_frame.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)

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
        self.head_canvas.pack()

        self.head_pose_label = tk.Label(self.head_pose_viewer_frame, text="Head Pose Estimation")
        self.head_pose_label.pack()

        self.hand_canvas = tk.Canvas(self.hand_gesture_viwer_frame,
                                     width=self.frame_width * 0.2,
                                     height=self.frame_height * 0.2)
        self.hand_canvas.pack()

        self.command_label = tk.Label(self.command_frame, text="Command Detected:")
        self.command_label.grid(row=0, column=0)

        self.hand_gesture_label = tk.Label(self.hand_gesture_viwer_frame,
                                           text="Hand Gesture Recognition")
        self.hand_gesture_label.pack()

        self.gesture_name_label = tk.Label(self.command_frame,
                                           font="Courier 18 bold",
                                           text="",
                                           width=27,
                                           bg="#9b9b9b",
                                           fg="#084d6e")
        self.gesture_name_label.grid(row=0, column=1)

        self.check_boxes_frame = tk.Label(self.controls_and_info_frame)
        self.check_boxes_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.follow_head_var = tk.IntVar()
        self.follow_head_var.set(1)
        self.follow_hand_var = tk.IntVar()

        cb1 = tk.Checkbutton(self.check_boxes_frame,
                             text='Follow Head',
                             onvalue=1,
                             offvalue=0,
                             variable=self.follow_head_var,
                             command=self.follow_selection)
        cb1.pack(side=tk.TOP, anchor=tk.W)

        cb2 = tk.Checkbutton(self.check_boxes_frame,
                             text='Follow Hand',
                             onvalue=1,
                             offvalue=0,
                             variable=self.follow_hand_var,
                             command=self.follow_selection)
        cb2.pack(side=tk.TOP, anchor=tk.W)

        self.detection_time_frame = tk.Frame(self.controls_and_info_frame)
        self.detection_time_frame.pack(side=tk.LEFT, anchor=tk.NW)

        self.detection_time_label = tk.Label(self.detection_time_frame,
                                             text="Detection Time:")
        self.detection_time_label.pack()

        self.timer_label = tk.Label(self.detection_time_frame,
                                    font="Courier 45 bold",
                                    text="0",
                                    height=1,
                                    width=2,
                                    bg="#9b9b9b",
                                    fg="#084d6e")
        self.timer_label.pack()

        self.zoom_indicator_frame = tk.Frame(self.controls_and_info_frame)
        self.zoom_indicator_frame.pack(side=tk.LEFT, anchor=tk.N, padx=15)

        self.zoom_indicator_label = tk.Label(self.zoom_indicator_frame,
                                             text="Zoom:")
        self.zoom_indicator_label.pack()

        self.zoom_label = tk.Label(self.zoom_indicator_frame,
                                    font="Courier 45 bold",
                                    text="1x",
                                    height=1,
                                    width=2,
                                    bg="#9b9b9b",
                                    fg="#084d6e")
        self.zoom_label.pack()

        self.dropdown_selectors_frame = tk.Frame(self.controls_and_info_frame)
        self.dropdown_selectors_frame.pack(side=tk.LEFT, anchor=tk.N)

        self.camera = tk.StringVar()
        self.camera.set("Select Camera")
        self.cameras_index = ["Cam 1", "Cam 2"]

        self.camera_selector_dropdown = tk.OptionMenu(self.dropdown_selectors_frame,
                                                      self.camera,
                                                      *self.cameras_index,
                                                      command=self.camera_selection)
        self.camera_selector_dropdown.config(width=17)
        self.camera_selector_dropdown.pack(pady=17)

        self.com_port = tk.StringVar()
        self.com_port.set("Select COM Port")
        self.com_port_index = ["COM 1", "COM 2"]

        self.com_port_dropdown = tk.OptionMenu(self.dropdown_selectors_frame,
                                               self.com_port,
                                               *self.com_port_index,
                                               command=self.com_port_selection)
        self.com_port_dropdown.config(width=17)
        self.com_port_dropdown.pack()




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

    def follow_selection(self):
        """Updates selection values."""
        if self.follow_head_var.get() == 1:
            self.follow_hand_var.set(0)

        if self.follow_hand_var.get() == 1:
            self.follow_head_var.set(0)

    def camera_selection(self, selection):
        """Selects the camera index."""
        print("Selection:", selection)

    def com_port_selection(self, selection):
        """Selects the COM port index."""
        print("Selection:", selection)



if __name__ == "__main__":
    from multiprocessing import Queue, Pipe
    place_holder_1 = Queue()
    place_holder_2 = Queue()
    place_holder_3 = Queue()
    recv_place_holder_4, send_place_holder_4  = Pipe()
    place_holder_5 = Queue()

    gui = GuiApplication(place_holder_1,
                         place_holder_2,
                         place_holder_3,
                         send_place_holder_4,
                         place_holder_5)
