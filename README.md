# Classroom Autonomous Cam System

This project was developed as part of my undergraduate thesis for a bachelor's degree in electrical engineering with an emphasis on control and automation at the Federal University of Campina Grande. The text of the work (in Brazilian Portuguese) can be accessed here: [LYANG LEME DE MEDEIROS -MONOGRAFIA-ENGENHARIA ELÉTRICA-CEEI (2022)](https://dspace.sti.ufcg.edu.br/bitstream/riufcg/33232/1/LYANG%20LEME%20DE%20MEDEIROS%20-MONOGRAFIA-ENGENHARIA%20EL%C3%89TRICA-CEEI%20%282022%29.pdf)

## GENERAL OBJECTIVES
The general objective of this work was to develop a system that facilitates a teacher's ability to teach classes to students both in person and remotely, allowing them to move and be automatically followed by the camera, and to control the zoom using hand gestures.

## SPECIFIC OBJECTIVES
In order to map and organize the necessary steps to achieve the proposed general objective, the following specific objectives were defined:
- To study and evaluate open-source libraries based on machine learning and/or artificial intelligence focused on video processing;
- To develop an algorithm capable of detecting and tracking people, recognizing faces and also hand gestures;
- To develop a software-controlled mechanical system that allows the movement of a camera.

## System Overview Diagram
<img width="722" height="452" alt="Basic-Diagram drawio" src="https://github.com/user-attachments/assets/c2bce611-9f63-441e-9b8c-2d62ee54a366" />

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

### Screenshot 
<img width="945" height="532" alt="image" src="https://github.com/user-attachments/assets/4509b016-47db-4625-b7ea-33b8b31cff68" />



