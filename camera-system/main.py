"""Main module for Classroom Autonomus Camera System."""

from multiprocessing import freeze_support
from camera_system.frame_server import FrameServer
from camera_system.process_manager import ProcessManager
from camera_system.gui_application import GuiApplication
from camera_system.frame_processing import FrameProcessing
from camera_system.serial_messenger import SerialMessenger
from camera_system.camera_controller import CameraController
from camera_system.frame_acquisition import FrameAcquisition

CAMERA_INDEX = 1

def acquirer_worker(frames_queue):
    """Proxy function for creating the frame acquisition process. If a proxy
    function is not used for this process the pickle module raises an exception
     because of opencv."""

    camera = FrameAcquisition(CAMERA_INDEX)
    camera.open_camera()
    camera.acquirer_worker(frames_queue)


def serial_messeger_worker(pipe_connection):
    """Proxy function for serial communication."""

    serial_messeger = SerialMessenger()
    serial_messeger.serial_worker(pipe_connection)


if __name__ == '__main__':
    freeze_support()

    frame_server = FrameServer(frame_step=1)
    frame_processor = FrameProcessing()
    process_manager = ProcessManager(1)
    camera_controller = CameraController()

    process_manager.set_serial_communication_process(serial_messeger_worker)
    process_manager.serial_communication_process.start()

    process_manager.set_acquirer_process(acquirer_worker)
    process_manager.acquirer_process.start()

    process_manager.set_frame_server_process(frame_server.start_server)
    process_manager.server_process.start()

    process_manager.set_head_pose_estimation_process(frame_processor.head_pose_estimation)
    process_manager.head_pose_estimation_process.start()

    process_manager.set_hand_gesture_recognition_process(frame_processor.hand_gesture_recognition)
    process_manager.hand_gesture_recognition_process.start()

    process_manager.set_hand_command_receiver_process(camera_controller.hand_command_receiver)
    process_manager.hand_command_receiver_process.start()

    following_state_pipes = (process_manager.recv_following_state1,
                            process_manager.send_following_state2)
    gui = GuiApplication(process_manager.queue_raw_frame_server_output,
                         process_manager.queue_head_pose_estimation_output,
                         process_manager.queue_hand_gesture_recognition_output,
                         process_manager.queue_processed_frames_output,
                         process_manager.recv_gesture_label,
                         process_manager.recv_zoom_gui,
                         process_manager.queue_gesture_duration,
                         following_state_pipes)

    process_manager.close_all_pipes()
    process_manager.close_all_queues()
    process_manager.terminate_all_processes()
