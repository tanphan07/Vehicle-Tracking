# Vehicle Tracking using YOLOV8 + Bytetrack

This project is built by combining [YOLOv8](https://github.com/ultralytics/ultralytics) and [Bytetrack](https://github.com/ifzhang/ByteTrack) for multiple vehicle tracking in the streets.

## <div align="center">Documentation</div>
## News 
- [✅] Add tracking by detection.
- Coming soon: Add tracking by segmentation using SAM( Segment Anything Model).
## Installation
Step 1: Create a conda enviroment
```shell
conda create -n tracking python=3.8
```
Step 2: Clone the repository
```shell
git clone https://github.com/tanphan07/Vehicle-Tracking.git
```
Step 3: Install package dependencies
```shell
cd Vehicle-Tracking
pip3 install -r requirements.txt
```

## Dataset Preparation
In this project, I have used COCO dataset to train the model. You can download the dataset from [here](https://cocodataset.org/#download) then filter the vehicle image in the dataset or you can use this file to automating download COCO 
```shell
python3 download_dataset.py
``` 
After downloading the dataset, you need to convert the dataset to YOLO format. You can use `prepare_dataset.py` file to convert the dataset to YOLO format (change the path to your dataset path).

## Training
In this project, I have used YOLOv8 to train the model. You can follow the instruction in [this](https://github.com/ultralytics/ultralytics) to train the model. After training, you will get the weight file in `runs/train/exp/weights/best.pt`. For the better performance, I used TensorRT to optimize the model. You can convert the weight file to TensorRT engine following the instruction in YOLOv8 repository. I have train a model and convert it to TensorRT engine. You can download the engine [here](https://drive.google.com/drive/u/0/folders/1otvyp9atEXNSgFZwhz2HhxBu-2xXXALB).


## Tracking
After getting the TensorRT engine, you can use `tracking.py` to track the vehicle in the video. You can change the path to your video in the file.
```shell
python3 tracking.py
```
You can change your detection model by changing `detection.py` file. This project allow you use another detection model for testing.

## Result
Here is a demo video of the project. The original video [here](https://www.youtube.com/shorts/Hrk0FhylkEU). And the result video [here](https://www.youtube.com/shorts/eow1W7SLPxA).