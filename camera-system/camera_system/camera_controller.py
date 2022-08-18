"""Module that controls the camera."""

from multiprocessing import Lock

import queue
import time
import cv2

class CameraController:
    """Class that controls the camera."""

    MINIMUM_ZOOM = 1
    MAXIMUM_ZOOM = 5
    ZOOM_STEP = 1.02
    BLOCKED_TIME = 6

    follow_head = True
    cropped_width = 1280
    cropped_height = 720
    pad_x = 0
    pad_y = 0
    current_zoom = 1
    block_time = 0
    blocked = False
    position_received = False
    indicator_coordinates_received = False

    def __init__(self):
        pass

    def hand_command_receiver(self, queue_input, queue_output, comand_pipe,
                              head_position_pipe, serial_pipe_connection,
                              indicator_pipe, zoom_pipe, following_state_pipes, virtual_cam_queue):
        """Receives and executes the hand commands."""

        recv_following_state, send_following_state = following_state_pipes

        while True:
            command = ""
            head_angle = 0
            nose_coordinates = (0, 0)
            indicator_x = 0
            indicator_y = 0

            if head_position_pipe.poll():
                self.position_received = True
                head_angle, nose_coordinates = head_position_pipe.recv()

            if indicator_pipe.poll():
                indicator_x, indicator_y = indicator_pipe.recv()
                self.indicator_coordinates_received = True

            if not serial_pipe_connection.poll():
                if self.follow_head:
                    if self.position_received:
                        serial_pipe_connection.send((head_angle, nose_coordinates))
                elif not self.follow_head and self.indicator_coordinates_received:
                    self.indicator_coordinates_received = False
                    serial_pipe_connection.send((0, (indicator_x // 5, indicator_y // 5)))

                self.position_received = False

            if comand_pipe.poll():
                command = comand_pipe.recv()

            frame = queue_input.get()
            height, width = frame.shape[:2]

            if command.find("Zoom") != -1:
                self.set_zoom_parameters(height, width, command)
                self.send_zoom_value_to_gui(zoom_pipe)

            if recv_following_state.poll():
                follow_state = recv_following_state.recv()
                if follow_state == 1:
                    self.follow_head = True
                elif follow_state == 0:
                    self.follow_head = False

            if command == "Toggle Following" and not self.blocked:
                self.blocked = True
                self.block_time = time.time()
                self.follow_head = not self.follow_head

            if not send_following_state.poll():
                send_following_state.send(self.follow_head)

            frame = frame[self.pad_y:self.pad_y + self.cropped_height,
                          self.pad_x:self.pad_x + self.cropped_width]

            frame = cv2.resize(frame, (height, width), interpolation=cv2.INTER_CUBIC)

            if self.blocked:
                self.blocked = (time.time() - self.block_time) < self.BLOCKED_TIME

            try:
                queue_output.put_nowait(frame)
            except queue.Full:
                pass
            try:
                virtual_cam_queue.put_nowait(frame)
            except queue.Full:
                pass

    def set_zoom_parameters(self,  height, width, command):
        """Sets the zoom in or zoom out parameters."""

        self.current_zoom = 1 / (self.cropped_width / width)

        if command == "Zoom In":

            if self.current_zoom <= self.MAXIMUM_ZOOM:
                self.cropped_width = int(self.cropped_width // self.ZOOM_STEP)
                self.cropped_height = int(self.cropped_height // self.ZOOM_STEP)

        elif command == "Zoom Out":

            if self.current_zoom >= self.MINIMUM_ZOOM:
                self.cropped_width = int(self.cropped_width * self.ZOOM_STEP)
                self.cropped_height = int(self.cropped_height * self.ZOOM_STEP)

        self.pad_x = (width - self.cropped_width) // 2
        self.pad_y = (height - self.cropped_height) // 2

        self.pad_x = max(self.pad_x, 0)
        self.pad_y = max(self.pad_y, 0)

    def send_zoom_value_to_gui(self, zoom_pipe):
        """Sends the current zoom value to the GUI."""

        if not zoom_pipe.poll():
            zoom_pipe.send(self.current_zoom)
