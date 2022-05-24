import serial
import time

MILISSECOND = 1/1e3
WAIT_SERIAL_CONNECTION = 2

def write_data(driver, x):
    driver.write(bytes(str(x), 'utf-8'))
    time.sleep(2.1 * MILISSECOND)
    try:
        data = (driver.readline()).decode()
    except ValueError:
        data = -1
    return data


def str_padding(str_value):
    
    signal = "+"
    
    if (not str_value.isnumeric()):
        signal = str_value[0]
        str_value = str_value[1:]

    padding = '0' * (4 - len(str_value))

    str_value = signal + padding + str_value

    return str_value

def command_builder(phi_value, theta_value):
    str_phi = str(phi_value)
    str_theta = str(theta_value)

    str_phi = str_padding(str_phi)
    str_theta = str_padding(str_theta)

    command = str_phi + "/" + str_theta + "!"
    return command


def send_command(driver, phi_value, theta_value):
    
    payload = command_builder(phi_value, theta_value)
    resp = write_data(driver, payload)

    if (payload == str(resp).replace("\r\n", "")):
        return True
    
    return False


def get_driver():
    
    driver = serial.Serial(port='COM3', baudrate=115200, timeout=.1)
    time.sleep(WAIT_SERIAL_CONNECTION)
    return driver

if __name__ == '__main__':
    
    motor_driver = get_driver()
    send_command(motor_driver, 200, 0)
    send_command(motor_driver, -400, 0)
    send_command(motor_driver, 200, 0)