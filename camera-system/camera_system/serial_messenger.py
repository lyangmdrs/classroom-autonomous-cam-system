"""Module that manages serial communication."""
import time
import serial

class SerialMessenger:
    """Class that manages serial communication."""

    WAIT_SERIAL_CONNECTION = 2
    HEAD_ANGLE_THRESHOLD = 10
    MILISSECOND = 1/1e3
    FRAME_HEIGHT = 720
    FRAME_WIDTH = 1280
    X_STEP = 2
    Y_STEP = 3

    def __init__(self, debug=False):

        self.debug = debug

        try:
            self.driver = serial.Serial(port='COM3', baudrate=115200, timeout=.1)
        except serial.SerialException:
            self.driver = None
        else:
            time.sleep(self.WAIT_SERIAL_CONNECTION)

            if self.debug:
                print("Serial Connected!")

    def string_padding(self, value):
        """Add the correct number of zeros to the string to build a valid message."""

        signal = "+"

        if not value.isnumeric():
            signal = value[0]
            value = value[1:]

        padding = '0' * (4 - len(value))

        return signal + padding + value

    def build_command_string(self, pan_value, tilt_value):
        """Builds a valid string to command the Pan-Tilt driver."""

        command_separator = "/"
        command_terminator = "!"

        str_pan_value = self.string_padding(str(pan_value))
        str_tilt_value = self.string_padding(str(tilt_value))

        return str_pan_value + command_separator + str_tilt_value + command_terminator

    def send_command_and_get_response(self, command):
        """Sends the serial command via serial."""

        response = None
        try:
            self.driver.write(bytes(str(command), "utf-8"))
            time.sleep(2.1 * self.MILISSECOND)
        except AttributeError:
            return response

        try:
            response = (self.driver.readline()).decode()
        except ValueError:
            response = None
        return response

    def serial_worker(self, pipe_connection):
        """Manages the serial communication."""

        while True:

            head_angle, nose_coordinates = pipe_connection.recv()
            x_distance = int((self.FRAME_WIDTH * 0.2) // 2 - nose_coordinates[0]) // self.X_STEP
            # y_distance = int(nose_coordinates[1] - (self.FRAME_HEIGHT * 0.2) // 2) // self.Y_STEP

            command = self.build_command_string(0, x_distance)

            text = "looking forward"
            if head_angle < -self.HEAD_ANGLE_THRESHOLD:
                text = "looking left"
            elif head_angle > self.HEAD_ANGLE_THRESHOLD:
                text = "looking right"

            if self.debug:
                print(f"Head is {text}!")
                print(f"Head angle: {head_angle}")
                print(f"Command: {command}")

            response = self.send_command_and_get_response(command)

            if response and self.debug:
                print(f"Response: {response}")
