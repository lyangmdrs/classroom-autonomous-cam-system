"""Main module for Classroom Autonomus Camera System."""
import csv
import time
import queue
import multiprocessing
import tkinter as tk
import itertools
import copy
import PIL.Image
import PIL.ImageTk
import numpy as np
import mediapipe as mp
import serial
import cv2
import win32pipe
import win32file
import pywintypes

from collections import deque
from collections import Counter
from model import KeyPointClassifier
from model import PointHistoryClassifier
from multiprocessing import Pipe, Process, Queue

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


class FrameProcessing:
    """Class to group frame processing methods."""

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    last_gesture = ""
    elapsed_time = 0
    initial_time = 0
    last_time = 0
    is_counting = False
    scaled_width = 1280
    scaled_height = 720
    pad_x = 0
    pad_y = 0
    zoom_step = 1.2
    current_zoom = 0
    maximum_zoom = 2

    def __init__(self):
        pass

    def head_pose_estimation(self, queue_input, queue_output, pipe_connection):
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

                    if not pipe_connection.poll():
                        pipe_connection.send((y_coordenate, nose_2d))

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

    def hand_gesture_recognition(self, queue_input, queue_output,
                                 pipe_connection, zoom_pipe):
        """Recognizes hand gestures that eventually appear in frames received
        by the input queue, draws landmakrs and the nose direction vector.
        At the end, it attaches the edited frame to the output queue."""

        max_gestures_list_len = 15
        hand_gesture_list = deque(maxlen=max_gestures_list_len)

        hands = self.mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1)

        keypoint_classifier = KeyPointClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

        with open('model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

        while True:
            input_frame = queue_input.get()
            results = hands.process(input_frame)
            gesture_label = ""

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                    bounding_box = self.calculate_hand_bounding_box(input_frame, hand_landmarks)
                    landmark_list = self.calculate_hand_landmarks_list(input_frame, hand_landmarks)


                    cv2.rectangle(input_frame, (bounding_box[0], bounding_box[1]),
                                  (bounding_box[2], bounding_box[3]), (0, 0, 0), 1)
                    self.draw_hand_landmarks(input_frame,landmark_list)

                    pre_processed_landmark_list = self.pre_process_landmarks(landmark_list)

                    hand_gesture_id = keypoint_classifier(pre_processed_landmark_list)
                    hand_gesture_list.append(hand_gesture_id)
                    most_common_gesture = Counter(hand_gesture_list).most_common()

                    gesture_label = keypoint_classifier_labels[most_common_gesture[0][0]]
            else:
                self.last_gesture = ""

            if not pipe_connection.poll():
                pipe_connection.send(gesture_label)

            if gesture_label != "":
                if gesture_label == self.last_gesture:
                    if not self.is_counting:
                        self.initial_time = time.time()
                        self.is_counting = True
                    else:
                        self.last_time = time.time()
                else:
                    self.last_gesture = gesture_label
                    self.initial_time = 0
                    self.last_time = 0
                    self.is_counting = False

            if self.initial_time != 0:
                self.elapsed_time = self.last_time - self.initial_time

            if self.elapsed_time > 5:
                if self.last_gesture == "Zoom In":
                    if not zoom_pipe.poll():
                        zoom_pipe.send(self.last_gesture)
                elif self.last_gesture == "Zoom Out":
                    if not zoom_pipe.poll():
                        zoom_pipe.send(self.last_gesture)
            queue_output.put(input_frame)

    def calculate_hand_bounding_box(self, image, landmarks):
        """Calculates the points of a bounding box to the hands on the frame."""

        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calculate_hand_landmarks_list(self, image, landmarks):
        """Calculates the landmark points to the hands on the frame."""

        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def draw_hand_landmarks(self, image, points):
        """Draws the landmark points of hands on the frame."""

        if len(points) > 0:
            cv2.line(image, tuple(points[2]), tuple(points[3]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[2]), tuple(points[3]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[3]), tuple(points[4]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[3]), tuple(points[4]), (255, 255, 255), 2)

            cv2.line(image, tuple(points[5]), tuple(points[6]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[5]), tuple(points[6]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[6]), tuple(points[7]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[6]), tuple(points[7]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[7]), tuple(points[8]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[7]), tuple(points[8]), (255, 255, 255), 2)

            cv2.line(image, tuple(points[9]), tuple(points[10]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[9]), tuple(points[10]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[10]), tuple(points[11]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[10]), tuple(points[11]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[11]), tuple(points[12]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[11]), tuple(points[12]), (255, 255, 255), 2)

            cv2.line(image, tuple(points[13]), tuple(points[14]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[13]), tuple(points[14]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[14]), tuple(points[15]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[14]), tuple(points[15]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[15]), tuple(points[16]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[15]), tuple(points[16]), (255, 255, 255), 2)

            cv2.line(image, tuple(points[17]), tuple(points[18]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[17]), tuple(points[18]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[18]), tuple(points[19]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[18]), tuple(points[19]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[19]), tuple(points[20]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[19]), tuple(points[20]), (255, 255, 255), 2)

            cv2.line(image, tuple(points[0]), tuple(points[1]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[0]), tuple(points[1]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[1]), tuple(points[2]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[1]), tuple(points[2]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[2]), tuple(points[5]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[2]), tuple(points[5]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[5]), tuple(points[9]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[5]), tuple(points[9]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[9]), tuple(points[13]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[9]), tuple(points[13]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[13]), tuple(points[17]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[13]), tuple(points[17]), (255, 255, 255), 2)
            cv2.line(image, tuple(points[17]), tuple(points[0]), (0, 0, 0), 6)
            cv2.line(image, tuple(points[17]), tuple(points[0]), (255, 255, 255), 2)

        for index, landmark in enumerate(points):
            if index == 0:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def pre_process_landmarks(self, landmark_list):
        """Pre-processes the landmark values."""
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def apply_zoom(self, queue_input, queue_output, pipe_connection):
        """Applies zoom in or zoom out to the output frames."""

        while True:
            command = ""
            frame = queue_input.get()
            height, width = frame.shape[:2]

            if pipe_connection.poll():
                command = pipe_connection.recv()

            current_zoom = self.scaled_width // width

            if command == "Zoom In":

                if current_zoom <= 4:
                    self.scaled_width = int(self.scaled_width * self.zoom_step)
                    self.scaled_height = int(self.scaled_height * self.zoom_step)

            elif command == "Zoom Out":

                if current_zoom >= 1:
                    self.scaled_width = int(self.scaled_width // self.zoom_step)
                    self.scaled_height = int(self.scaled_height // self.zoom_step)

            self.pad_x = (self.scaled_width - width) // 2
            self.pad_y = (self.scaled_height - height) // 2

            self.pad_x = max(self.pad_x, 0)
            self.pad_y = max(self.pad_y, 0)

            self.scaled_width = max(self.scaled_width, width)
            self.scaled_height = max(self.scaled_height, height)

            frame = frame[self.pad_y:self.pad_y+self.scaled_height,
                          self.pad_x:self.pad_x+self.scaled_width]

            frame = cv2.resize(frame, (self.scaled_width, self.scaled_height),
                               interpolation=cv2.INTER_LINEAR)

            try:
                queue_output.put_nowait(frame)
            except queue.Full:
                pass


class FrameServer:
    """Class that manages the distribution of frames between the
    different processes of the autonomous camera system."""

    FRAME_RESIZE_FACTOR = 0.2
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    def __init__(self, frame_step=5):

        self.frame_step = frame_step

    def start_server(self, queue_raw_frame_server_input: Queue,
                     queue_raw_frame_server_output: Queue,
                     queue_head_pose_estimation_output: Queue,
                     queue_hand_gesture_recognition_output: Queue,
                     queue_processed_frame_output: Queue):
        """Starts frame server main loop."""

        frame_counter = 0

        while True:
            frame = queue_raw_frame_server_input.get()
            try:
                queue_raw_frame_server_output.put_nowait(frame)
            except queue.Full:
                pass
            try:
                queue_processed_frame_output.put_nowait(frame)
            except queue.Full:
                pass

            frame_counter = (frame_counter + 1) % self.frame_step
            head_pose_estimation_frame = cv2.resize(frame,
                                            (int(self.FRAME_WIDTH * self.FRAME_RESIZE_FACTOR),
                                            int(self.FRAME_HEIGHT * self.FRAME_RESIZE_FACTOR)),
                                            interpolation = cv2.INTER_AREA)

            if frame_counter == 0:
                try:
                    queue_head_pose_estimation_output.put_nowait(head_pose_estimation_frame)
                except queue.Full:
                    pass

                try:
                    queue_hand_gesture_recognition_output.put_nowait(frame)
                except queue.Full:
                    pass


class ProcessManager:
    """Class that manages the processes of the autonomous camera system."""

    _all_queues_ = []
    _all_processes_ = []

    def __init__(self, queues_size: int):

        self.queue_raw_frame_server_input = Queue(queues_size)
        self.queue_raw_frame_server_output = Queue(queues_size)

        self.queue_hand_gesture_recognition_input = Queue(queues_size)
        self.queue_hand_gesture_recognition_output = Queue(queues_size)

        self.queue_head_pose_estimation_input = Queue(queues_size)
        self.queue_head_pose_estimation_output = Queue(queues_size)

        self.queue_processed_frames_input = Queue(queues_size)
        self.queue_processed_frames_output = Queue(queues_size)

        self.server_process = Process()
        self.acquirer_process = Process()
        self.head_pose_estimation_process = Process()
        self.hand_gesture_recognition_process = Process()
        self.head_pose_pipe_connection_process = Process()
        self.serial_communication_process = Process()
        self.zoom_process = Process()

        self.recv_connection, self.send_connection = Pipe()
        self.recv_gesture_label, self.send_gesture_label = Pipe()
        self.recv_zoom_process, self.send_zoom_process = Pipe()

        self._all_queues_ = [self.queue_raw_frame_server_input,
                            self.queue_raw_frame_server_output,
                            self.queue_hand_gesture_recognition_input,
                            self.queue_hand_gesture_recognition_output,
                            self.queue_head_pose_estimation_input,
                            self.queue_head_pose_estimation_output,
                            self.queue_processed_frames_input,
                            self.queue_processed_frames_output]

    def set_acquirer_process(self, acquirer_target):
        """Configures the process for acquiring frames."""

        self.acquirer_process = Process(target=acquirer_target,
                                        args=(self.queue_raw_frame_server_input,))
        self._all_processes_.append(self.acquirer_process)

    def set_frame_server_process(self, frame_server_target):
        """Configures the process for the frame server."""

        self.server_process = Process(target=frame_server_target,
                                      args=(self.queue_raw_frame_server_input,
                                            self.queue_raw_frame_server_output,
                                            self.queue_head_pose_estimation_input,
                                            self.queue_hand_gesture_recognition_input,
                                            self.queue_processed_frames_input,))
        self._all_processes_.append(self.server_process)

    def set_head_pose_estimation_process(self, head_pose_estimation_target):
        """Configures the frame processing process for head pose estimation."""

        self.head_pose_estimation_process = Process(target=head_pose_estimation_target,
                                                    args=(self.queue_head_pose_estimation_input,
                                                          self.queue_head_pose_estimation_output,
                                                          self.send_connection,))
        self._all_processes_.append(self.head_pose_estimation_process)

    def set_hand_gesture_recognition_process(self, hand_gesture_recognition_target):
        """Configures the frame processing process for manual gesture recognition."""

        self.hand_gesture_recognition_process = Process(target=hand_gesture_recognition_target,
                                                    args=(self.queue_hand_gesture_recognition_input,
                                                    self.queue_hand_gesture_recognition_output,
                                                    self.send_gesture_label,
                                                    self.send_zoom_process,))
        self._all_processes_.append(self.hand_gesture_recognition_process)

    def set_head_pose_pipe_connection_process(self, head_pose_pipe_connection_target):
        """Configures the head pose pipe connection process for manual gesture recognition."""

        self.head_pose_pipe_connection_process = Process(target=head_pose_pipe_connection_target)
        self._all_processes_.append(self.head_pose_pipe_connection_process)

    def set_serial_communication_process(self, serial_communication_target):
        """Configures the serial communication process."""

        self.serial_communication_process = Process(target=serial_communication_target,
                                                    args=(self.recv_connection,))
        self._all_processes_.append(self.serial_communication_process)

    def set_zoom_process(self, zoom_target):
        """Configrues the process that applies zoom in and zoom out."""
        self.zoom_process = Process(target=zoom_target,
                                    args=(self.queue_processed_frames_input,
                                    self.queue_processed_frames_output,
                                    self.recv_zoom_process,))
        self._all_processes_.append(self.zoom_process)

    def close_all_queues(self):
        """Terminates all frame queues used in processes."""

        for _queue in self._all_queues_:
            _queue.close()

    def terminate_all_processes(self):
        """Terminates all processes."""

        for _process in self._all_processes_:
            if _process.is_alive():
                _process.terminate()

    def close_all_pipes(self):
        """Closes all pipes."""
        self.recv_connection.close()
        self.send_connection.close()


class SerialMessenger:
    """Class that manages serial communication."""

    WAIT_SERIAL_CONNECTION = 2
    HEAD_ANGLE_THRESHOLD = 10
    MILISSECOND = 1/1e3
    FRAME_HEIGHT = 720
    FRAME_WIDTH = 1280
    X_STEP = 2
    Y_STEP = 3

    def __init__(self):

        try:
            self.driver = serial.Serial(port='COM3', baudrate=115200, timeout=.1)
        except serial.SerialException:
            self.driver = None
        else:
            time.sleep(self.WAIT_SERIAL_CONNECTION)
            print("Serial Connected!")

    def string_padding(self, value):
        """Add the correct number of zeros to the string to build a valid message."""

        signal = "+"

        if not value.isnumeric():
            signal = value[0]
            value = value[1:]

        padding = '0' * (4 - len(value))

        return signal + padding + value

    def build_command_string(self, pan_value, tilt_value):
        """Builds a valid string to command the Pan-Tilt driver."""

        command_separator = "/"
        command_terminator = "!"

        str_pan_value = self.string_padding(str(pan_value))
        str_tilt_value = self.string_padding(str(tilt_value))

        return str_pan_value + command_separator + str_tilt_value + command_terminator

    def send_command_and_get_response(self, command):
        """Sends the serial command via serial."""

        response = None
        try:
            self.driver.write(bytes(str(command), "utf-8"))
            time.sleep(2.1 * self.MILISSECOND)
        except AttributeError:
            return response

        try:
            response = (self.driver.readline()).decode()
        except ValueError:
            response = None
        return response



    def serial_worker(self, pipe_connection):
        """Manages the serial communication."""

        while True:

            head_angle, nose_coordinates = pipe_connection.recv()
            x_distance = int((self.FRAME_WIDTH * 0.2) // 2 - nose_coordinates[0]) // self.X_STEP
            y_distance = int(nose_coordinates[1] - (self.FRAME_HEIGHT * 0.2) // 2) // self.Y_STEP

            command = self.build_command_string(0, x_distance)

            text = "looking forward"
            if head_angle < -self.HEAD_ANGLE_THRESHOLD:
                text = "looking left"
            elif head_angle > self.HEAD_ANGLE_THRESHOLD:
                text = "looking right"

            # print(f"Head is {text}!")
            # print(f"Head angle: {head_angle}")
            # print(f"Command: {command}")

            response = self.send_command_and_get_response(command)

            if response:
                print(f"Response: {response}")


def acquirer_proxy(frames_queue):
    """Proxy function for creating the frame acquisition process. If a proxy
    function is not used for this process the pickle module raises an exception
     because of opencv."""

    camera = FrameAcquisition()
    camera.open_camera()
    camera.acquirer_worker(frames_queue)


def serial_messeger_worker(pipe_connection):
    """Proxy function for serial communication."""

    serial_messeger = SerialMessenger()
    serial_messeger.serial_worker(pipe_connection)


if __name__ == '__main__':
    multiprocessing.freeze_support()

    frame_server = FrameServer(frame_step=1)
    frame_processor = FrameProcessing()
    process_manager = ProcessManager(1)

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
