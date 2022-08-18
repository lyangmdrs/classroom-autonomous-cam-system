"""Module that send frames to the virtual camera."""

import queue
import socket
import cv2
import numpy as np

class VirtualCamServer:
    """Class that send frame to the virtual camera."""
    __HOST = "127.0.0.1"
    __PORT = 15015
    __IMAGE_WIDTH = 1280
    __IMAGE_HEIGHT = 720
    __IMAGE_DEPTH = 3
    __IMAGE_SIZE = __IMAGE_WIDTH * __IMAGE_HEIGHT

    def __init__(self):
        pass

    def virtual_camera_worker(self, input_frames_queue):
        """Virtual camera worker."""
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as frame_server:
                frame_server.bind((self.__HOST, self.__PORT))
                frame_server.listen()
                print("Waiting Connection...")
                conn, addr = frame_server.accept()
                with conn:
                    print(f"Connected by {addr}")
                    initial_frame = np.reshape(np.repeat(155,
                                               self.__IMAGE_SIZE*self.__IMAGE_DEPTH),
                                               (self.__IMAGE_HEIGHT,
                                               self.__IMAGE_WIDTH,
                                               self.__IMAGE_DEPTH))

                    frame = np.asarray(initial_frame, np.uint8)
                    frame = cv2.putText(frame, "STARTING...",
                                        (150, 450), 0, 6,
                                        (8,77,110), 6, 16, False)

                    while True:
                        try:
                            frame = input_frames_queue.get_nowait()
                        except queue.Empty:
                            pass
                        else:
                            frame = cv2.flip(frame, 0)
                            frame = cv2.flip(frame, 1)
                            frame = cv2.resize(frame, (1280, 720))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            asc = frame.tobytes()
                            try:
                                conn.sendall(asc)
                            except Exception as error:
                                print("Error:", error)
                                conn.close()
                                frame_server.shutdown(socket.SHUT_RD)
                                frame_server.close()
