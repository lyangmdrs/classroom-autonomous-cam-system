"""Module that creates a graphic user interface."""
import tkinter as tk
import queue
import numpy as np
import cv2
import PIL.Image
import PIL.ImageTk
import wmi

class GuiApplication:
    """Class that creates a graphic user interface."""

    __WINDOW_NAME = "Classroom Autonomous Camera System"
    __WIDTH = 1040
    __HEIGHT = 510
    __WINDOW_GEOMETRY = str(__WIDTH) + "x" + str(__HEIGHT)
    __WINDOW_UPDATE_DELAY = 1
    __MAIN_FRAME_RESIZE_FACTOR = 0.25
    __AUX_FRAME_RESIZE_FACTOR = 0.1
    __FRAME_WIDTH = 1280
    __FRAME_HEIGHT = 720
    __FRAME_DEPTH = 3
    __FOUR_CHAR_ZOOM_FONT = "Courier 22 bold"

    hand_gesture_name = ""

    initial_frame = np.reshape(np.repeat(155, __FRAME_HEIGHT*__FRAME_WIDTH*__FRAME_DEPTH),
                               (__FRAME_HEIGHT, __FRAME_WIDTH, __FRAME_DEPTH))


    raw_frame = np.asarray(initial_frame, np.uint8)
    raw_frame = cv2.putText(raw_frame, "STARTING...", (150, 450), 0, 6,  (8,77,110), 6, 16, False)
    head_pose_frame = cv2.resize(raw_frame,
                                 (__FRAME_WIDTH // 5, __FRAME_HEIGHT // 5),
                                 interpolation=3)
    hand_pose_frame = raw_frame
    processed_frame = raw_frame

    enumerator = wmi.WMI()

    def __init__(self, raw_frame_queue, head_pose_queue, hand_pose_queue, output_frame_queue,
                 hand_gesture_pipe, zoom_pipe, queue_gesture_duration, following_state_pipes,
                 cam_index_queue, serial_port_queue):

        self.window = tk.Tk()
        self.window.title(self.__WINDOW_NAME)
        self.window.geometry(self.__WINDOW_GEOMETRY)

        self.zoom_pipe = zoom_pipe
        self.raw_frame_queue = raw_frame_queue
        self.head_pose_queue = head_pose_queue
        self.hand_pose_queue = hand_pose_queue
        self.cam_index_queue = cam_index_queue
        self.processed_queue = output_frame_queue
        self.hand_gesture_pipe = hand_gesture_pipe
        self.serial_port_queue = serial_port_queue
        self.queue_gesture_duration = queue_gesture_duration
        self.recv_folloing_state, self.send_following_state = following_state_pipes

        self.main_viewer_frame = tk.Frame(self.window, height=self.__HEIGHT // 2)
        self.main_viewer_frame.pack(side=tk.TOP, ipady=10, fill=tk.X)

        self.debug_viewer_frame = tk.Frame(self.window, height=self.__HEIGHT // 2)
        self.debug_viewer_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.head_pose_viewer_frame = tk.Frame(self.debug_viewer_frame, 
                                               width=self.__FRAME_WIDTH * 0.2 + 10)
        self.head_pose_viewer_frame.pack(side=tk.LEFT, ipadx=5,
                                         fill=tk.Y, expand=True)

        self.hand_gesture_viwer_frame = tk.Frame(self.debug_viewer_frame,
                                                 width=self.__FRAME_WIDTH * 0.2 + 10)
        self.hand_gesture_viwer_frame.pack(side=tk.LEFT, ipadx=5,
                                           fill=tk.Y, expand=True)

        self.controls_and_info_frame = tk.Frame(self.debug_viewer_frame, width=self.__WIDTH // 2 - 5)
        self.controls_and_info_frame.pack(side=tk.LEFT, anchor=tk.W, fill=tk.BOTH, expand=True)

        self.command_frame = tk.Frame(self.controls_and_info_frame, width=self.__WIDTH // 2 - 10)
        self.command_frame.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)

        self.input_viewer_frame = tk.Frame(self.main_viewer_frame)
        self.input_viewer_frame.pack(side=tk.LEFT)

        self.output_viewer_frame = tk.Frame(self.main_viewer_frame)
        self.output_viewer_frame.pack(side=tk.RIGHT)

        self.input_cava = tk.Canvas(self.input_viewer_frame, width=self.__FRAME_WIDTH * 0.4,
                                     height=self.__FRAME_HEIGHT * 0.4)
        self.input_cava.pack(side=tk.TOP)

        self.input_label = tk.Label(self.input_viewer_frame, text="Input")
        self.input_label.pack(side=tk.BOTTOM)

        self.output_canva = tk.Canvas(self.output_viewer_frame, width=self.__FRAME_WIDTH * 0.4,
                                     height=self.__FRAME_HEIGHT * 0.4)
        self.output_canva.pack(side=tk.TOP)

        self.output_label = tk.Label(self.output_viewer_frame, text="Output")
        self.output_label.pack(side=tk.BOTTOM)

        self.head_canvas = tk.Canvas(self.head_pose_viewer_frame,
                                     width=self.__FRAME_WIDTH * 0.2,
                                     height=self.__FRAME_HEIGHT * 0.2)
        self.head_canvas.pack()

        self.head_pose_label = tk.Label(self.head_pose_viewer_frame, text="Head Pose Estimation")
        self.head_pose_label.pack()

        self.hand_canvas = tk.Canvas(self.hand_gesture_viwer_frame,
                                     width=self.__FRAME_WIDTH * 0.2,
                                     height=self.__FRAME_HEIGHT * 0.2)
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

        self.follow_var = tk.IntVar()
        self.follow_var.set(1)

        rb1 = tk.Radiobutton(self.check_boxes_frame,
                             text='Follow Head',
                             value = 1,
                             variable=self.follow_var,
                             command=self.follow_selection)
        rb1.pack(side=tk.TOP, anchor=tk.W)

        rb2 = tk.Radiobutton(self.check_boxes_frame,
                             text='Follow Hand',
                             value=0,
                             variable=self.follow_var,
                             command=self.follow_selection)
        rb2.pack(side=tk.TOP, anchor=tk.W)

        self.detection_time_frame = tk.Frame(self.controls_and_info_frame)
        self.detection_time_frame.pack(side=tk.LEFT, anchor=tk.NW)

        self.detection_time_label = tk.Label(self.detection_time_frame,
                                             text="Detection Time:")
        self.detection_time_label.pack()

        self.timer_label = tk.Label(self.detection_time_frame,
                                    font="Courier 35 bold",
                                    text="0s",
                                    height=1,
                                    width=2,
                                    bg="#9b9b9b",
                                    fg="#084d6e")
        self.timer_label.pack(ipady=6, ipadx=12)

        self.zoom_indicator_frame = tk.Frame(self.controls_and_info_frame)
        self.zoom_indicator_frame.pack(side=tk.LEFT, anchor=tk.N, padx=15)

        self.zoom_indicator_label = tk.Label(self.zoom_indicator_frame,
                                             text="Zoom:")
        self.zoom_indicator_label.pack()

        self.zoom_label = tk.Label(self.zoom_indicator_frame,
                                    font=self.__FOUR_CHAR_ZOOM_FONT,
                                    text="1.0x",
                                    height=2,
                                    width=4,
                                    bg="#9b9b9b",
                                    fg="#084d6e")
        self.zoom_label.pack(ipadx=5)

        self.dropdown_selectors_frame = tk.Frame(self.controls_and_info_frame)
        self.dropdown_selectors_frame.pack(side=tk.LEFT, anchor=tk.N)

        self.camera = tk.StringVar()
        self.camera.set("Select Camera")
        self.cameras_index = ["Enumerating Cameras"]
        self.enumerate_cameras()

        self.camera_selector_dropdown = tk.OptionMenu(self.dropdown_selectors_frame,
                                                      self.camera,
                                                      *self.cameras_index,
                                                      command=self.camera_selection)
        self.camera_selector_dropdown.config(width=17)
        self.camera_selector_dropdown.pack(pady=17)

        self.com_port = tk.StringVar()
        self.com_port.set("Select COM Port")
        self.com_port_index = []
        self.enumerate_serial_devices()

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

        if self.hand_gesture_pipe.poll():
            self.hand_gesture_name = self.hand_gesture_pipe.recv()
            self.gesture_name_label.configure(text=self.hand_gesture_name)

        input_canva_frame = cv2.resize(self.raw_frame,
                                      (int(self.__FRAME_WIDTH * 0.4),
                                      int(self.__FRAME_HEIGHT * 0.4)),
                                      interpolation = cv2.INTER_AREA)

        self.input_canva_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(input_canva_frame))
        self.input_cava.create_image(0, 0, image=self.input_canva_frame, anchor = tk.NW)

        heap_pose_frame = PIL.Image.fromarray(self.head_pose_frame)
        self.face_canva_frame = PIL.ImageTk.PhotoImage(image=heap_pose_frame)
        self.head_canvas.create_image(0, 0, image=self.face_canva_frame, anchor = tk.NW)

        hand_canva_frame = cv2.resize(self.hand_pose_frame,
                                      (int(self.__FRAME_WIDTH * 0.2),
                                      int(self.__FRAME_HEIGHT * 0.2)),
                                      interpolation = cv2.INTER_AREA)
        self.hand_canva_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(hand_canva_frame))
        self.hand_canvas.create_image(0, 0, image=self.hand_canva_frame, anchor = tk.NW)

        output_canva_frame = cv2.resize(self.processed_frame,
                                        (int(self.__FRAME_WIDTH * 0.4),
                                        int(self.__FRAME_HEIGHT * 0.4)),
                                        interpolation = cv2.INTER_AREA)
        self.output_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(output_canva_frame))
        self.output_canva.create_image(0, 0, image=self.output_frame, anchor = tk.NW)

        self.update_gesture_duration()
        self.update_zoom_indicator()
        self.update_folloing_state()

        self.window.after(self.delay, self.update)

    def follow_selection(self):
        """Sends following states to camera controller."""
        if not self.send_following_state.poll():
            self.send_following_state.send(self.follow_var.get())

    def update_folloing_state(self):
        """Receives the folloing states from camera controller."""
        if self.recv_folloing_state.poll():
            follow_state = self.recv_folloing_state.recv()
            if follow_state is True:
                self.follow_var.set(1)
            elif follow_state is False:
                self.follow_var.set(0)

    def camera_selection(self, selection):
        """Selects the camera index."""
        while True:
            try:
                self.cam_index_queue.put_nowait(self.cameras_index.index(selection))
                self.camera_selector_dropdown.config(state=tk.DISABLED)
                break
            except queue.Full:
                pass

    def com_port_selection(self, selection:str):
        """Selects the COM port index."""
        selection = selection.split()
        for part in selection:
            if "COM" in part:
                selection = part.strip("()")

        if ("COM" in selection) is False:
            return

        while True:
            try:
                self.serial_port_queue.put_nowait(selection)
                self.com_port_dropdown.config(state=tk.DISABLED)
                break
            except queue.Full:
                pass

    def update_zoom_indicator(self):
        """Updtates zoom indicator on GUI."""
        zoom_value = 0
        if self.zoom_pipe.poll():
            zoom_value = self.zoom_pipe.recv()
            if zoom_value > 1:
                zoom_value = round(zoom_value, 1)
                self.zoom_label.config(text=str(zoom_value) + "x")

    def update_gesture_duration(self):
        """Updates gesture duration indicator on GUI."""
        try:
            gesture_duration = self.queue_gesture_duration.get_nowait()
            if gesture_duration >= 0 and gesture_duration <= 5.2:
                self.timer_label.config(text=str(int(gesture_duration)) + 's')
                self.timer_label.config(fg="#084d6e")
            elif gesture_duration > 5.2:
                self.timer_label.config(text=str("OK"))
                self.timer_label.config(fg="#1a5c45")
        except queue.Empty:
            pass

    def enumerate_cameras(self):
        """Enumerates the connected cameras."""
        query = "Select * From Win32_USBControllerDevice"
        devices = []
        for device in self.enumerator.query(query):
            device_class = device.Dependent.PNPClass
            device_name = device.Dependent.Name
            if device_class.upper() in ('CAMERA', 'IMAGE'):
                devices.append(device_name)
        self.cameras_index = devices

    def enumerate_serial_devices(self):
        """Enumerates the connected serial devices."""
        query = "Select * From Win32_USBControllerDevice"
        devices = []
        for device in self.enumerator.query(query):
            device_class = device.Dependent.PNPClass
            device_name = device.Dependent.Name
            if device_class.upper() == "PORTS":
                devices.append(device_name)
        if len(devices) > 0:
            self.com_port_index = devices
        else:
            self.com_port_index = ["NO DEVICES FOUND"]


if __name__ == "__main__":
    from multiprocessing import Queue, Pipe
    place_holder_1 = Queue()
    place_holder_2 = Queue()
    place_holder_3 = Queue()
    place_holder_5 = Queue()
    recv_place_holder_4, send_place_holder_4  = Pipe()
    recv_place_holder_6, send_place_holder_6  = Pipe()
    recv_place_holder_7, send_place_holder_7  = Pipe()
    place_holder_8 = Queue()
    place_holder_9 = Queue()

    gui = GuiApplication(place_holder_1,
                         place_holder_2,
                         place_holder_3,
                         place_holder_5,
                         send_place_holder_4,
                         recv_place_holder_6,
                         place_holder_5,
                         (recv_place_holder_4,
                         send_place_holder_6),
                         place_holder_8,
                         place_holder_9)
