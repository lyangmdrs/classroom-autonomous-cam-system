"""Module to group frame processing methods."""
from collections import deque
from collections import Counter

import csv
import copy
import time
import queue
import itertools
import cv2
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

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
                                 pipe_connection, command_pipe):
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
                if not command_pipe.poll():
                    command_pipe.send(self.last_gesture)

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
