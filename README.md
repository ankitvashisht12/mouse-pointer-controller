# Computer Pointer Controller

Computer Pointer Controller app is an Edge AI application which is used to control the movement of mouse pointer by the direction of eye gaze and pose of person's head. This application requires webcam or video feed as input and by using the Gaze detection model, it will move the mouse curson using `pyautogui` library.

## Project Set Up and Installation

## Setup

> NOTE : You'll need to install the [Openvion toolkit](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) and required libraries from `requirements.txt` file.

- Step 1 : Download this project
- Step 2 : Initialize the openVINO environment using the following command

`source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5`

> Note : For windows, you may need to refer the docs

- Step 3 : Download the required models by using OpenVINO model downloader
1. Face Detection model 

```py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001" 
```

2. Facial Landmark detection model

```py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```

3. Head pose Detection model 

```py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```

4. Gaze Estimation model
	
```py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

> Note : Move the models into root directory of the project in `models` directory.

## Demo

To run the basic demo :

- Open Terminal / CMD and change the directory to root directory of the project
- Use the following command to run the app

```sh
python src/app.py -m1 "Path of face detection xml file" \
	-m2 "Path of facial landmark detection xml file" \ 
	-m3 "Path of head pose detection xml file" \
	-m4 "Path of Gaze Estimation xml file" \
	-d CPU \
	-i video \
	-p bin/demo.mp4 \
  -v fd lm hp ge
```

* Check out the [Video demo of app here](https://www.youtube.com/watch?v=k8LnWRhvQho)

### Command Line Options 


```
usage: app.py [-h] -m1 FACE_DETECTION -m2 LANDMARK_DETECTION -m3
              HEAD_POSE_DETECTION -m4 GAZE_DETECTION [-d DEVICE]
              [-e EXTENTION] -i INPUT_FILE [-p INPUT_PATH]
              [-v VISUALIZE [VISUALIZE ...]] [-prob PROB]

Mouse Pointer Controller using eye gaze

optional arguments:
  -h, --help            show this help message and exit
  -m1 FACE_DETECTION, --face_detection FACE_DETECTION
                        Path to face detection
  -m2 LANDMARK_DETECTION, --landmark_detection LANDMARK_DETECTION
                        Path to landmark detection model
  -m3 HEAD_POSE_DETECTION, --head_pose_detection HEAD_POSE_DETECTION
                        Path to head pose detection model
  -m4 GAZE_DETECTION, --gaze_detection GAZE_DETECTION
                        Path to gaze detection model
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -e EXTENTION, --extention EXTENTION
                        Specify CPU extention
  -i INPUT_FILE, --input_file INPUT_FILE
                        Specify input file type
  -p INPUT_PATH, --input_path INPUT_PATH
                        Specify input file path
  -v VISUALIZE [VISUALIZE ...], --visualize VISUALIZE [VISUALIZE ...]
                        Specify the flags from fd, fld, hp, ge like --flags fd
                        hp fld (Seperate each flag by space)for see the
                        visualization of different model outputs of each
                        frame,fd for Face Detection, ld for Facial Landmark
                        Detectionhp for Head Pose Estimation, ge for Gaze
                        Estimation.
  -prob PROB, --prob PROB
                        threshod probability
```


## Directory Structure

```bash
 ├── mouse-pointer-controller/         
    ├── images/                   # supported images for README.md
    ├── bin/                      # Demo video's
    ├── src/                      # contain all the app.py and all the source code.   
    ├── models/                   # model files
    ├── requirements.txt        
    ├── app.log
    └── README.md
```

## Documentation

*Model Documentations* : 

1. [Face Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html) 
2. [Facial Landmark detection model](https://docs.openvinotoolkit.org/latest/_models_intel_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html)
3. [Head pose Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
4. [Gaze Estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)


## Benchmarks

#### Benchmark Results on i5-5200U CPU

**Performance Analysis of FP32 Precision Models (in seconds )**

| Name | Loading FP32 | FPS ( in seconds ) | Avg. Inference time FP32 | Total Inference time FP32 |
| --- | --- | --- | --- |--- | 
| Face detection | 0.46175241470336914 | 25.59700246966908 | 0.03906707440392446 | 2.304957389831543 | 
| Facial Landmark detection | 0.9716916084289551 | 281.5110027120001 | 0.0035522590249271718 | 0.20958328247070312 | 
| Head Pose detection | 0.21136951446533203 | 312.7201978448859 | 0.003197746761774612 | 0.18866705894470215 | 
| Gaze Estimation | 0.525824785232544 | 221.20195508454276 | 0.00452075570316638 | 0.2667245864868164 | 


**Performance Analysis of FP16 Precision Models (in seconds )**

| Name | Loading FP16 | FPS ( in seconds ) | Avg. Inference time FP16 | Total Inference time FP16 |
| --- | --- | --- | --- |--- | 
| Face detection | -- | -- | -- | -- | 
| Facial Landmark detection | 1.7589049339294434 | 284.23515358400977 | 0.0035182136600300415 | 0.20757460594177246 | 
| Head Pose detection | 0.4001595973968506 | 393.37309921441084 | 0.0025421158742096463 | 0.14998483657836914 | 
| Gaze Estimation | 0.29756903648376465 | 229.9613478535207 | 0.0043485568741620595 | 0.2565648555755615 | 



## Results

As we can clearly observe the total inference time is decreased and FPS value increased in FP16. Thus FP16 gives best result in my case than FP32 because lower precision models can process fast calculations. Normally FP32 gives best result with CPU in terms of accuracy than FP16 as lowering precision of the model decreases the accuracy of the model.
