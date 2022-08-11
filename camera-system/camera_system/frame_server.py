"""Module that manages the distribution of frames between the
    different processes of the autonomous camera system."""
from multiprocessing import Queue
import queue
import cv2

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
