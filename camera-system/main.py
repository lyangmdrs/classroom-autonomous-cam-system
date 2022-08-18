"""Main module for Classroom Autonomus Camera System."""

from multiprocessing import freeze_support
from camera_system.frame_server import FrameServer
from camera_system.process_manager import ProcessManager
from camera_system.gui_application import GuiApplication
from camera_system.frame_processing import FrameProcessing
from camera_system.serial_messenger import SerialMessenger
from camera_system.camera_controller import CameraController
from camera_system.frame_acquisition import FrameAcquisition
from camera_system.virtual_camera_server import VirtualCamServer

def acquirer_worker(cam_index_queue, frames_queue):
    """Proxy function for creating the frame acquisition process. If a proxy
    function is not used for this process the pickle module raises an exception
     because of opencv."""

    camera = FrameAcquisition()
    camera.get_camera_index(cam_index_queue)
    camera.open_camera()
    camera.acquirer_worker(frames_queue)


def serial_messeger_worker(pipe_connection, serial_port_queue):
    """Proxy function for serial communication."""

    serial_messeger = SerialMessenger()
    serial_messeger.get_serial_port(serial_port_queue)
    serial_messeger.connect_serial()
    serial_messeger.serial_worker(pipe_connection)


if __name__ == '__main__':
    freeze_support()

    process_manager = ProcessManager(1)
    frame_processor = FrameProcessing()
    camera_controller = CameraController()
    frame_server = FrameServer(frame_step=1)
    virtual_camera = VirtualCamServer()

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

    process_manager.set_virtual_camera_process(virtual_camera.virtual_camera_worker)
    process_manager.virtual_camera_process.start()

    following_state_pipes = (process_manager.recv_following_state1,
                             process_manager.send_following_state2)
    gui = GuiApplication(process_manager.queue_raw_frame_server_output,
                         process_manager.queue_head_pose_estimation_output,
                         process_manager.queue_hand_gesture_recognition_output,
                         process_manager.queue_processed_frames_output,
                         process_manager.recv_gesture_label,
                         process_manager.recv_zoom_gui,
                         process_manager.queue_gesture_duration,
                         following_state_pipes,
                         process_manager.queue_camera_index,
                         process_manager.queue_serial_port)

    process_manager.close_all_pipes()
    process_manager.close_all_queues()
    process_manager.terminate_all_processes()
