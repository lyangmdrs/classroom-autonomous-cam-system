# Classroom Autonomous Cam System

## Pan-Tilt Driver

### PlatformIO Command Lines

To work with the PlatformIO command lines, the PlatformIO Core (CLI) must be installed. To install, follow the instructions at:

- [PlatformIO Core Local Download](https://docs.platformio.org/en/latest//core/installation.html#local-download-mac-linux-windows)
- [PlatformIO Windows Instalation](https://docs.platformio.org/en/latest//core/installation.html#windows)

To execute the commands browse to the directory ***classroom-autonomous-cam-system\pan-tilt-driver***

- Verify the source code: ```pio run```
- Load the source code to the board: ```pio run --target uno```
- Open a serial monitor: ```pio device monitor```
- Run tests: ```pio test -e uno```
