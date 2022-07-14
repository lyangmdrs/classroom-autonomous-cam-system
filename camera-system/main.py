"""Main module for Classroom Autonomus Camera System."""
import multiprocessing
from multiprocessing import Lock, Process, Queue
import tkinter as tk
import queue
import PIL.Image
import PIL.ImageTk
import mediapipe as mp
import numpy as np
import time
import cv2
import win32pipe
import win32file
import pywintypes


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

    raw_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)
    head_pose_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)
    hand_pose_frame = np.zeros((frame_width, frame_height, frame_depth), np.uint8)

    def __init__(self, raw_frame_queue, head_pose_queue, hand_pose_queue):
        self.window = tk.Tk()
        self.window.title(self._WINDOW_NAME)

        self.raw_frame_queue = raw_frame_queue
        self.head_pose_queue = head_pose_queue
        self.hand_pose_queue = hand_pose_queue

        self.main_viewer_frame = tk.Frame(self.window, width=self.frame_width * 0.6,
                                          height=self.frame_height * 0.6)
        self.main_viewer_frame.pack(side=tk.RIGHT)

        self.debug_viewer_frame = tk.Frame(self.window, height=self.frame_height * 0.4)
        self.debug_viewer_frame.pack(side=tk.LEFT)

        self.main_canvas = tk.Canvas(self.main_viewer_frame, width=self.frame_width * 0.5,
                                     height=self.frame_height * 0.5)
        self.main_canvas.pack(side=tk.TOP)

        self.main_stream_label = tk.Label(self.main_viewer_frame, text="Input")
        self.main_stream_label.pack(side=tk.BOTTOM)

        self.face_canvas = tk.Canvas(self.debug_viewer_frame, width=self.frame_width * 0.2,
                                     height=self.frame_height * 0.2)
        self.face_canvas.pack(side=tk.TOP)

        self.main_stream_label = tk.Label(self.debug_viewer_frame, text="Head Pose Estimation")
        self.main_stream_label.pack(side=tk.TOP)

        self.hand_canvas = tk.Canvas(self.debug_viewer_frame, width=self.frame_width * 0.2,
                                     height=self.frame_height * 0.2)
        self.hand_canvas.pack(side=tk.TOP)

        self.main_stream_label = tk.Label(self.debug_viewer_frame, text="Hand Gesture Recognition")
        self.main_stream_label.pack(side=tk.TOP)

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

        main_canva_frame = cv2.resize(self.raw_frame,
                                      (int(self.frame_width * 0.5),
                                      int(self.frame_height * 0.5)),
                                      interpolation = cv2.INTER_AREA)

        self.main_canva_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(main_canva_frame))
        self.main_canvas.create_image(0, 0, image=self.main_canva_frame, anchor = tk.NW)

        face_canva_frame = cv2.resize(self.head_pose_frame,
                                      (int(self.frame_width * 0.2),
                                      int(self.frame_height * 0.2)),
                                      interpolation = cv2.INTER_AREA)
        self.face_canva_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(face_canva_frame))
        self.face_canvas.create_image(0, 0, image=self.face_canva_frame, anchor = tk.NW)

        hand_canva_frame = cv2.resize(self.hand_pose_frame,
                                      (int(self.frame_width * 0.2),
                                      int(self.frame_height * 0.2)),
                                      interpolation = cv2.INTER_AREA)
        self.hand_canva_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(hand_canva_frame))
        self.hand_canvas.create_image(0, 0, image=self.hand_canva_frame, anchor = tk.NW)

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
        self.camera.open(self.camera_index)

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


class FrameProcessing:
    """Class to group frame processing methods."""

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    def __init__(self):
        pass

    def connect_head_pose_pipe(self):
        """Waits for pipe connection."""

        self.head_pose_pipe = win32pipe.CreateNamedPipe(r'\\.\pipe\Serial',
                                                        win32pipe.PIPE_ACCESS_DUPLEX,
                                                        win32pipe.PIPE_TYPE_MESSAGE |
                                                        win32pipe.PIPE_READMODE_MESSAGE |
                                                        win32pipe.PIPE_WAIT,
                                                        1, 65536, 65536, 0, None)
        print("Pipe Criado!")
        win32pipe.ConnectNamedPipe(self.head_pose_pipe, None)
        print("Pipe Connectado!")
        time.sleep(5)
        win32file.CloseHandle(self.head_pose_pipe)

    def head_pose_estimation(self, queue_input, queue_output):
        """Estimates the position of one of the people's head that eventually
        appears in the frames received by the input queue, draws the landmakrs
        and the nose direction vector. At the end it appends the edited frame
        to the output queue."""

        with self.mp_holistic.Holistic(min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5) as holistic:
            while True:
                input_frame = queue_input.get()
                results = holistic.process(input_frame)

                img_h, img_w, _ = input_frame.shape
                face_3d = []
                face_2d = []

                if results.face_landmarks:
                    for idx, landmark in enumerate(results.face_landmarks.landmark):
                        if idx == 33 or idx == 263 or \
                            idx == 1 or idx == 61 or \
                            idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (landmark.x * img_w, landmark.y * img_h)

                            x_coordenate = int(landmark.x * img_w)
                            y_coordenate = int(landmark.y * img_h)

                            face_2d.append([x_coordenate, y_coordenate])
                            face_3d.append([x_coordenate, y_coordenate, landmark.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    focal_length = 1 * img_w

                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d,
                                                         cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    x_coordenate = angles[0] * 360
                    y_coordenate = angles[1] * 360
                    _ = angles[2] * 360

                    point1 = (int(nose_2d[0]), int(nose_2d[1]))
                    point2 = (int(nose_2d[0] + y_coordenate * 10),
                          int(nose_2d[1] - x_coordenate * 10))

                    cv2.line(input_frame, point1, point2, (255, 0, 0), 3)

                self.mp_drawing.draw_landmarks(input_frame,
                                               results.face_landmarks,
                                               self.mp_holistic.FACEMESH_TESSELATION,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mp_drawing_styles
                                               .get_default_face_mesh_tesselation_style())
                queue_output.put(input_frame)

    def hand_gesture_recognition(self, queue_input, queue_output):
        """Recognizes hand gestures that eventually appear in frames received
        by the input queue, draws landmakrs and the nose direction vector.
        At the end, it attaches the edited frame to the output queue.

        NOT IMPLEMENTED"""
        #TODO: implement this function.
        while True:
            input_frame = queue_input.get()
            output_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
            queue_output.put(output_frame)


class FrameServer:
    """Class that manages the distribution of frames between the
    different processes of the autonomous camera system."""

    def __init__(self, frame_step=5):

        self.frame_step = frame_step

    def start_server(self, queue_raw_frame_server_input: Queue,
                     queue_raw_frame_server_output: Queue,
                     queue_head_pose_estimation_input: Queue,
                     queue_hand_gesture_recognition_input: Queue):
        """Starts frame server main loop."""

        frame_counter = 0
        while True:
            frame = queue_raw_frame_server_input.get()
            try:
                queue_raw_frame_server_output.put_nowait(frame)
                frame_counter = (frame_counter + 1) % self.frame_step
                if frame_counter == 0:
                    queue_head_pose_estimation_input.put_nowait(frame)
                    queue_hand_gesture_recognition_input.put_nowait(frame)
            except queue.Full:
                continue


class ProcessManager:
    """Class that manages the processes of the autonomous camera system."""

    _all_queues_ = []

    def __init__(self, queues_size: int):

        self.queue_raw_frame_server_input = Queue(queues_size)
        self.queue_raw_frame_server_output = Queue(queues_size)

        self.queue_hand_gesture_recognition_input = Queue(queues_size)
        self.queue_hand_gesture_recognition_output = Queue(queues_size)

        self.queue_head_pose_estimation_input = Queue(queues_size)
        self.queue_head_pose_estimation_output = Queue(queues_size)

        self.server_process = Process()
        self.acquirer_process = Process()
        self.head_pose_estimation_process = Process()
        self.hand_gesture_recognition_process = Process()
        self.head_pose_pipe_connection_process = Process()

        self._all_queues_ = [self.queue_raw_frame_server_input,
                            self.queue_raw_frame_server_output,
                            self.queue_hand_gesture_recognition_input,
                            self.queue_hand_gesture_recognition_output,
                            self.queue_head_pose_estimation_input,
                            self.queue_head_pose_estimation_output,]

    def set_acquirer_process(self, acquirer_target):
        """Configures the process for acquiring frames."""

        self.acquirer_process = Process(target=acquirer_target,
                                        args=(self.queue_raw_frame_server_input,))

    def set_frame_server_process(self, frame_server_target):
        """Configures the process for the frame server."""

        self.server_process = Process(target=frame_server_target,
                                      args=(self.queue_raw_frame_server_input,
                                            self.queue_raw_frame_server_output,
                                            self.queue_head_pose_estimation_input,
                                            self.queue_hand_gesture_recognition_input,))

    def set_head_pose_estimation_process(self, head_pose_estimation_target):
        """Configures the frame processing process for head pose estimation."""

        self.head_pose_estimation_process = Process(target=head_pose_estimation_target,
                                                    args=(self.queue_head_pose_estimation_input,
                                                          self.queue_head_pose_estimation_output,))

    def set_hand_gesture_recognition_process(self, hand_gesture_recognition_target):
        """Configures the frame processing process for manual gesture recognition."""

        self.hand_gesture_recognition_process = Process(target=hand_gesture_recognition_target,
                                                    args=(self.queue_hand_gesture_recognition_input,
                                                    self.queue_hand_gesture_recognition_output,))

    def set_head_pose_pipe_connection_process(self, head_pose_pipe_connection_target):
        """Configures the head pose pipe connection process for manual gesture recognition."""

        self.head_pose_pipe_connection_process = Process(target=head_pose_pipe_connection_target)

    def close_all_queues(self):
        """Terminates all frame queues used in processes."""

        for _queue in self._all_queues_:
            _queue.close()

    def terminate_all_processes(self):
        """Terminates all processes."""

        self.acquirer_process.terminate()
        self.server_process.terminate()
        self.head_pose_estimation_process.terminate()
        self.hand_gesture_recognition_process.terminate()


def acquirer_proxy(frames_queue):
    """Proxy function for creating the frame acquisition process. If a proxy
    function is not used for this process the pickle module raises an exception
     because of opencv."""

    camera = FrameAcquisition()
    camera.open_camera()
    camera.acquirer_worker(frames_queue)

def pipe_client(lock):
    """Head Pose pipe test"""

    quit_pipe = False

    while not quit_pipe:
        try:
            handle = win32file.CreateFile(r'\\.\pipe\Serial',
                                        win32file.GENERIC_READ |
                                        win32file.GENERIC_WRITE,
                                        0, None, 
                                        win32file.OPEN_EXISTING,
                                        0, None)

            resp = win32file.ReadFile(handle, 64*1024)

        except pywintypes.error as error:
            if error.args[0] == 2:
                print("no pipe, trying again in a sec")
                time.sleep(1)
            elif error.args[0] == 109:
                print("broken pipe")
                quit_pipe = True


if __name__ == '__main__':
    multiprocessing.freeze_support()

    locker = Lock()

    frame_server = FrameServer(frame_step=10)
    frame_processor = FrameProcessing()
    process_manager = ProcessManager(3)

    process_manager.set_acquirer_process(acquirer_proxy)
    process_manager.acquirer_process.start()

    process_manager.set_frame_server_process(frame_server.start_server)
    process_manager.server_process.start()

    process_manager.set_head_pose_pipe_connection_process(frame_processor.connect_head_pose_pipe)
    process_manager.head_pose_pipe_connection_process.start()

    process_manager.set_head_pose_estimation_process(frame_processor.head_pose_estimation)
    process_manager.head_pose_estimation_process.start()

    process_manager.set_hand_gesture_recognition_process(frame_processor.hand_gesture_recognition)
    process_manager.hand_gesture_recognition_process.start()

    pipe_client(locker)

    gui = GuiApplication(process_manager.queue_raw_frame_server_output,
                         process_manager.queue_head_pose_estimation_output,
                         process_manager.queue_hand_gesture_recognition_output)

    process_manager.close_all_queues()
    process_manager.terminate_all_processes()
