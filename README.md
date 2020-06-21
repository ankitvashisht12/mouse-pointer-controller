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
	1. **Face Detection model** 
``` py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001" 
```
	2. **Facial Landmark detection model**
```py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```

	3. **Head pose Detection model** :

```py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```

	4. **Gaze Estimation model** :
	
```py
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

> Note : Move the models into root directory of the project in `models` directory.

## Demo

To run the basic demo :

- Open Terminal / CMD and change the directory to root directory of the project
- Use the following command to run the app

```py
python src/app.py -m1 "Path of face detection xml file" \
	-m2 "Path of facial landmark detection xml file" \ 
	-m3 "Path of head pose detection xml file" \
	-m4 "Path of Gaze Estimation xml file" \
	-d "device type (CPU for cpu and GPU for gpu and HETERO:FPGA,CPU for FPGA)" \
	-i "input type (cam for webcam and video for video file)" \
	-p "path to video file if input type is video"
```

## Documentation

**Model Documentations** : 

1. [Face Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html) 
2. [Facial Landmark detection model](https://docs.openvinotoolkit.org/latest/_models_intel_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html)
3. [Head pose Detection model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
4. [Gaze Estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
