"""Module that controls the camera."""
import queue
import time
import cv2


class CameraController:
    """Class that controls the camera."""

    MINIMUM_ZOOM = 1
    MAXIMUM_ZOOM = 5
    ZOOM_STEP = 1.05
    BLOCK_TIME = 6

    follow_head = True
    cropped_width = 1280
    cropped_height = 720
    pad_x = 0
    pad_y = 0
    current_zoom = 0
    block_time = 0
    blocked = False
    position_received = False

    def __init__(self):
        pass

    def hand_command_receiver(self, queue_input, queue_output, comand_pipe,
                              head_position_pipe, serial_pipe_connection):
        """Receives and executes the hand commands."""

        while True:
            command = ""
            head_angle = 0
            nose_coordinates = (0, 0)

            if head_position_pipe.poll():
                self.position_received = True
                head_angle, nose_coordinates = head_position_pipe.recv()

            if not serial_pipe_connection.poll() and self.follow_head and self.position_received:
                serial_pipe_connection.send((head_angle, nose_coordinates))
                self.position_received = False

            if comand_pipe.poll():
                command = comand_pipe.recv()

            frame = queue_input.get()
            height, width = frame.shape[:2]

            if command.find("Zoom") != -1:
                self.set_zoom_parameters(height, width, command)

            if command == "Stop Following" and not self.blocked:
                self.blocked = True
                self.block_time = time.time()
                self.follow_head = not self.follow_head

            frame = frame[self.pad_y:self.pad_y + self.cropped_height,
                          self.pad_x:self.pad_x + self.cropped_width]

            frame = cv2.resize(frame, (height, width), interpolation=cv2.INTER_CUBIC)

            if self.blocked:
                self.blocked = (time.time() - self.block_time) < 5

            try:
                queue_output.put_nowait(frame)
            except queue.Full:
                pass

    def set_zoom_parameters(self,  height, width, command):
        """Sets the zoom in or zoom out parameters."""

        current_zoom = 1 / (self.cropped_width / width)

        if command == "Zoom In":

            if current_zoom <= self.MAXIMUM_ZOOM:
                self.cropped_width = int(self.cropped_width // self.ZOOM_STEP)
                self.cropped_height = int(self.cropped_height // self.ZOOM_STEP)

        elif command == "Zoom Out":

            if current_zoom >= self.MINIMUM_ZOOM:
                self.cropped_width = int(self.cropped_width * self.ZOOM_STEP)
                self.cropped_height = int(self.cropped_height * self.ZOOM_STEP)

        self.pad_x = (width - self.cropped_width) // 2
        self.pad_y = (height - self.cropped_height) // 2

        self.pad_x = max(self.pad_x, 0)
        self.pad_y = max(self.pad_y, 0)
 