"""Main module for Classroom Autonomus Camera System."""
import queue
import multiprocessing
import tkinter as tk
import PIL.Image
import PIL.ImageTk
import numpy as np
import cv2

from camera_system import serial_messenger as sm
from camera_system import process_manager as pm
from camera_system import frame_server as fs
from camera_system import frame_processing as fp
class GuiApplication:
    """Crates a graphic user interface."""

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


class FrameAcquisition:
    """Class to manage frame acquisition."""

    _FRAME_WIDTH = 1280
    _FRAME_HEIGHT = 720

    def __init__(self, camera_index=0):

        self.camera = cv2.VideoCapture()
        self.camera_index = camera_index
        self.width = None
        self.height = None

    def open_camera(self):
        """"Opens the acquisition device."""
        self.camera.open(self.camera_index, cv2.CAP_DSHOW)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._FRAME_HEIGHT)

        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        """Gets a frame from the acquisition device and returns it."""
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return (None, None)

    def acquirer_worker(self, outuput_queue):
        """Loop for acquiring frames and filling the queue of frames received as a parameter."""

        frame = np.zeros((self._FRAME_WIDTH, self._FRAME_HEIGHT, 3),
                         np.uint8)

        while True:
            _, frame = self.get_frame()
            try:
                outuput_queue.put_nowait(frame)
            except queue.Full:
                continue


    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()




def acquirer_proxy(frames_queue):
    """Proxy function for creating the frame acquisition process. If a proxy
    function is not used for this process the pickle module raises an exception
     because of opencv."""

    camera = FrameAcquisition(1)
    camera.open_camera()
    camera.acquirer_worker(frames_queue)


def serial_messeger_worker(pipe_connection):
    """Proxy function for serial communication."""

    serial_messeger = sm.SerialMessenger()
    serial_messeger.serial_worker(pipe_connection)


if __name__ == '__main__':
    multiprocessing.freeze_support()

    frame_server = fs.FrameServer(frame_step=1)
    frame_processor = fp.FrameProcessing()
    process_manager = pm.ProcessManager(1)

    process_manager.set_serial_communication_process(serial_messeger_worker)
    process_manager.serial_communication_process.start()

    process_manager.set_acquirer_process(acquirer_proxy)
    process_manager.acquirer_process.start()

    process_manager.set_frame_server_process(frame_server.start_server)
    process_manager.server_process.start()

    process_manager.set_head_pose_estimation_process(frame_processor.head_pose_estimation)
    process_manager.head_pose_estimation_process.start()

    process_manager.set_hand_gesture_recognition_process(frame_processor.hand_gesture_recognition)
    process_manager.hand_gesture_recognition_process.start()

    process_manager.set_zoom_process(frame_processor.apply_zoom)
    process_manager.zoom_process.start()

    gui = GuiApplication(process_manager.queue_raw_frame_server_output,
                         process_manager.queue_head_pose_estimation_output,
                         process_manager.queue_hand_gesture_recognition_output,
                         process_manager.recv_gesture_label,
                         process_manager.queue_processed_frames_output)

    process_manager.close_all_pipes()
    process_manager.close_all_queues()
    process_manager.terminate_all_processes()
