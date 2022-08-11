"""Module that controls the camera."""
import queue
import cv2


class CameraController:
    """Class that controls the camera."""

    MINIMUM_ZOOM = 1
    MAXIMUM_ZOOM = 5
    ZOOM_STEP = 1.05

    follow_head = True
    cropped_width = 1280
    cropped_height = 720
    pad_x = 0
    pad_y = 0
    current_zoom = 0

    def __init__(self):
        pass

    def apply_zoom(self, queue_input, queue_output, pipe_connection):
        """Applies zoom in or zoom out to the output frames."""

        while True:
            command = ""
            frame = queue_input.get()
            height, width = frame.shape[:2]

            if pipe_connection.poll():
                command = pipe_connection.recv()

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

            frame = frame[self.pad_y:self.pad_y + self.cropped_height,
                          self.pad_x:self.pad_x + self.cropped_width]

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

            try:
                queue_output.put_nowait(frame)
            except queue.Full:
                pass