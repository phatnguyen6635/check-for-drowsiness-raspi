# <div align="center">Check for Drowsiness</div>

## Table of Contents

- [About](#about)
- [Structure](#structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
    - [`main.py`](#mainpy)

## About

This project focuses on detecting worker drowsiness during the final inspection stage of a manufacturing process. Using MediaPipe’s Face Landmarker with Blendshapes, the system extracts facial features — specifically the eye openness ratio — to estimate the worker’s level of alertness in real time.

A custom logic module was implemented to analyze the degree of eye openness and determine drowsiness status. The solution is optimized and deployed on Raspberry Pi 4, enabling real-time monitoring with efficient on-edge inference.

This project aims to improve workplace safety and productivity by providing an automated, lightweight, and deployable solution for human condition monitoring in industrial environments


## Folder Structure

```structure
.
├── configs
│   └── configs.yaml
├── docs
│   └── mediapipe_face_landmarks.txt
├── logs
├── models
│   └── face_landmarker_v2_with_blendshapes.task
├── README.md
├── requirements.txt
├── run.sh
├── setup.py
├── src
│   ├── __init__.py
│   └── utils.py
├── tools
│   ├── logs
│   │   └── app.log
│   └── main.py
└── voice
    ├── voice1.wav
    └── voice2.wav
```

<img width="1297" height="590" alt="image" src="https://github.com/user-attachments/assets/fa74c9df-383d-4dac-9c97-3e4a7fb4252c" />

This source directories are as follows:

**Executable Files in `tools`:**

- **`run.sh`** - Run the project in real time on a Raspberry Pi 4 device connected to a webcam.

**Source Directories:**

- **configs** - Contains all model and algorithm configurations.
- **docs** - Contains all related documents.
- **log** - Contains log frpom the project.
- **models** - Contains MediaPipe Tasks model package. 
- **src** - Contains helper functions.
- **tools** - Contains all needed scripts for run this project.
- **voice** - Contains all voice files used in the project.

### Installation

1. Clone project repo
    ```
    git clone git@github.com:HitechMVP/CheckForDrowsinessRaspi_MV.git
    ```

2. Set up a virtual environment
    
    This project requires Anaconda for execution. If you do not have Anaconda installed, please download it from [here](https://docs.anaconda.com/anaconda/install/)
    
    Create a virtual environment name `check-for-drowsiness`

    ``` 
    conda create -n check-for-drowsiness python==3.9.0
    ```

    Activating the virtual environment

    ```
    conda activate check-for-drowsiness
    ```
 
3. Install project dependencies

    ```
    cd .\CheckForDrowsinessRaspi_MV\  
    pip install -e .
    ```

### Usage

To inference model in realtime on Rasberry Pi 4

```
bash run.sh
```
