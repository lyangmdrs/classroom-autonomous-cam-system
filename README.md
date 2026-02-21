# Classroom Autonomous Cam System

This project was developed as part of my undergraduate thesis for a bachelor's degree in electrical engineering with an emphasis on control and automation at the Federal University of Campina Grande. The text of the work (in Brazilian Portuguese) can be accessed here: [LYANG LEME DE MEDEIROS -MONOGRAFIA-ENGENHARIA ELÉTRICA-CEEI (2022)](https://dspace.sti.ufcg.edu.br/bitstream/riufcg/33232/1/LYANG%20LEME%20DE%20MEDEIROS%20-MONOGRAFIA-ENGENHARIA%20EL%C3%89TRICA-CEEI%20%282022%29.pdf)

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

### GUI Execution

The projet is developed for Windows 10, with Python 3.8.
To run the project following the instructions in this document, it is necessary to install ```make``` on Windows 10.

- Install the requires python modules: ```make install```
- Run the Classroom Autonomis Camera System: ```make run```
