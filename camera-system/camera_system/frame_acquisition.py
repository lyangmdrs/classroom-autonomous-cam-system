"""Module to manage frame acquisition."""
import queue
import numpy as np
import cv2

class FrameAcquisition:
    """Class to manage frame acquisition."""

    __FRAME_WIDTH = 1280
    __FRAME_HEIGHT = 720
    __INVALID_INDEX = -1


    def __init__(self, camera_index = __INVALID_INDEX):

        self.camera = cv2.VideoCapture()
        self.camera_index = camera_index
        self.width = None
        self.height = None

    def open_camera(self):
        """"Opens the acquisition device."""

        self.camera.open(self.camera_index, cv2.CAP_DSHOW)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.__FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__FRAME_HEIGHT)

        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_camera_index(self, camera_index_queue):
        """Gets the camera index."""

        index = self.__INVALID_INDEX
        while True:
            try:
                index = camera_index_queue.get()
            except queue.Empty:
                pass

            if index != self.__INVALID_INDEX:
                self.camera_index = index
                break

    def get_frame(self):
        """Gets a frame from the acquisition device and returns it."""
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return (None, None)

    def acquirer_worker(self, outuput_queue):
        """Loop for acquiring frames and filling the queue of frames received as a parameter."""

        frame = np.zeros((self.__FRAME_WIDTH, self.__FRAME_HEIGHT, 3),
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
