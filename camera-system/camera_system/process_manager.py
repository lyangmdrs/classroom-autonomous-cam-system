"""Module that manages the processes of the autonomous camera system."""
from multiprocessing import Pipe, Process, Queue

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

        self.queue_gesture_duration = Queue(1)
        self.queue_camera_index = Queue(1)
        self.queue_serial_port = Queue(1)

        self.server_process = Process()
        self.acquirer_process = Process()
        self.head_pose_estimation_process = Process()
        self.hand_gesture_recognition_process = Process()
        self.head_pose_pipe_connection_process = Process()
        self.serial_communication_process = Process()
        self.hand_command_receiver_process = Process()
        self.update_zoom_gui_inidicator_process = Process()

        self.recv_indicator_coordinates, self.send_indicator_coordinates = Pipe()
        self.recv_following_state1, self.send_following_state1 = Pipe()
        self.recv_following_state2, self.send_following_state2 = Pipe()
        self.recv_head_position, self.send_head_position = Pipe()
        self.recv_gesture_label, self.send_gesture_label = Pipe()
        self.recv_serial_pipe, self.send_serial_pipe = Pipe()
        self.recv_zoom_gui, self.send_zoom_gui = Pipe()
        self.recv_command, self.send_command = Pipe()


        self._all_queues_ = [self.queue_raw_frame_server_input,
                            self.queue_raw_frame_server_output,
                            self.queue_hand_gesture_recognition_input,
                            self.queue_hand_gesture_recognition_output,
                            self.queue_head_pose_estimation_input,
                            self.queue_head_pose_estimation_output,
                            self.queue_processed_frames_input,
                            self.queue_processed_frames_output,
                            self.queue_gesture_duration]

    def set_acquirer_process(self, acquirer_target):
        """Configures the process for acquiring frames."""

        self.acquirer_process = Process(target=acquirer_target,
                                        args=(self.queue_camera_index,
                                        self.queue_raw_frame_server_input,))
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
                                                          self.send_head_position,))
        self._all_processes_.append(self.head_pose_estimation_process)

    def set_hand_gesture_recognition_process(self, hand_gesture_recognition_target):
        """Configures the frame processing process for manual gesture recognition."""

        self.hand_gesture_recognition_process = Process(target=hand_gesture_recognition_target,
                                                    args=(self.queue_hand_gesture_recognition_input,
                                                    self.queue_hand_gesture_recognition_output,
                                                    self.send_gesture_label,
                                                    self.send_command,
                                                    self.send_indicator_coordinates,
                                                    self.queue_gesture_duration,))
        self._all_processes_.append(self.hand_gesture_recognition_process)

    def set_serial_communication_process(self, serial_communication_target):
        """Configures the serial communication process."""

        self.serial_communication_process = Process(target=serial_communication_target,
                                                    args=(self.recv_serial_pipe,
                                                    self.queue_serial_port))
        self._all_processes_.append(self.serial_communication_process)

    def set_hand_command_receiver_process(self, hand_command_receiver_target):
        """Configrues the process that receives hand commands."""
        self.hand_command_receiver_process = Process(target=hand_command_receiver_target,
                                                     args=(self.queue_processed_frames_input,
                                                     self.queue_processed_frames_output,
                                                     self.recv_command,
                                                     self.recv_head_position,
                                                     self.send_serial_pipe,
                                                     self.recv_indicator_coordinates,
                                                     self.send_zoom_gui,
                                                     (self.recv_following_state2,
                                                     self.send_following_state1),))
        self._all_processes_.append(self.hand_command_receiver_process)

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

        self.recv_indicator_coordinates.close()
        self.send_indicator_coordinates.close()
        self.recv_head_position.close()
        self.send_head_position.close()
        self.recv_gesture_label.close()
        self.send_gesture_label.close()
        self.recv_command.close()
        self.send_command.close()
