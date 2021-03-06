# Human Activity Recognition in Public Places
### Sean Campos, Praveen Reddy Kasireddy, Sam Shih, Fengyao Luo
Introducing a data pipeline with Deep Learning models to recognizing human actions in public places 

Research Paper [Human Activity Recognition](W251-Grp4-Paper_HumanActivityRecognition.pdf)

Presentation [Human Activity Recognition](W251-Grp4-Presentation_HumanActivityRecognition.pdf)

### Goal
- Build an end to end data pipeline to recognize human actions in the public places
- push metrics into cloud and stream it to a live dashboard

### Abstract

![Image of Park Image](https://github.com/fengyaoluo/Human-Activity-Monitoring-in-Public-Places/blob/main/images/Park%20image.PNG)

This project aims at collecting metrics pertaining to the activities of a person or a group of people based on sensor observation in the public realm, in order to inform better planning decisions. The application of activity recognition can be used in calculating the time span for human activities in public spaces such as parks. By leveraging the data, stakeholders can enhance the environment as well as the utilization of public facilities including benches, playgrounds, food areas, etc, as well as plan for maintenance, capital improvements and events.

In this project, we use an edge device, Jetson with a web camera to shoot a video, frame a person in the video, identify body parts, and recognize human postures. The pipeline is constructed with 2 main parts: Body Pose Detector and Action Recognition. The body pose detector used the Resnet18 as our backbone and trained it with the COCO dataset. It is able to pinpoint the coordinates of 18 key body parts and draw connections to each, forming a skeleton of 2D human pose. The action recognition used an LSTM to track a sequence of frames with pose vectors and trained with NTU RGB+D 60 dataset. The dataset includes 60 categories and we only picked 5 categories that were germain to our problem domain to train our model. The Identified activities include drink and eat, sit and squat, phone and talk, walk, selfie. We achieved a f1 score of 0.83 on the test dataset. Each category???s accuracy score has reached above 0.83 in the end.

In the ideal case, an activity is recognized regardless of the environment it is performed in or the performing person and data can be processed at a near real time pace. Instead of connecting a real time camera, we shoot a video which includes the actions to test our models, and output the measurable metrics, including actions, number of people, time span.

### Data Pipeline

![Image of Data Pipeline](https://github.com/fengyaoluo/Human-Activity-Monitoring-in-Public-Places/blob/main/images/data_pipeline.PNG)

#### Machine Learning at Edge

**Camera:**
- Stream video to the inference engine on the Edge Device


**Edge Device(Jetson):**
- Extract information about people and their body parts
- Perform  object tracking
- Detect activity
- Record the metrics


#### Phase 1: Body Part Detection

![Image of Phase1](https://github.com/fengyaoluo/Human-Activity-Monitoring-in-Public-Places/blob/main/images/phase1.PNG)

The first task in our pipeline is identifying people and their pose positions.

To do this:
The first step is to simply captures video from a webcam.

The second step uses a Resnet18 backbone to perform object recognition on human body parts.  

For the third step, a static list of associations that connect the body parts correctly and allows us to draw skeleton frames on top of people and a bounding box around each individual.  We then use an object tracking algorithm to track each person across a series of frames to build up a set of body part movement vectors that can be used for activity classification. 

A crucial limitation of object detection algorithms that they do not tell you if they???re detecting the same object in sequential frames.  To fulfill this requirement, we use the Hungarian algorithm, which can associate an object between one frame and the next. It calculates the intersection over union score, which is a value representing the amount of a bounding box that overlaps the previous frame.  In each frame, a matrix of all combinations of scores is calculated and the maximum IOU score between any two boxes above a given threshold is assumed to mean they are the same object.


#### Phase 2: Action Recognition Detection

![Image of Phase2](https://github.com/fengyaoluo/Human-Activity-Monitoring-in-Public-Places/blob/main/images/phase2.PNG)

After getting the data from the body part detection and human framing, we got a matrix with 18 key points per frame. In order to recognize the action, we took a window as 9 frame per sequence. We passed the key coordinates and labels from NTU training dataset into the LSTM model. Later I will show you the image frames from the NTU dataset. After training, this model output the predicted labels, and then we test them on our validation and test dataset. Finally, we output the confusion matrix and inference videos to measure the training results. 

![Image of NTU](https://github.com/fengyaoluo/Human-Activity-Monitoring-in-Public-Places/blob/main/images/NTU.png)

The bigger the movement of the key point, the larger information it includes. That is how model knows what kind of the change of key points is important when we try to classify actions. 


### Evaluation Results

![Image of Phase2](https://github.com/fengyaoluo/Human-Activity-Monitoring-in-Public-Places/blob/main/images/wandb.PNG)

We did grid search and tried different sets of parameters and ended up using image size 368 by 368, IoU is 0.5, area is medium for body part detection, and the layers dimension as 128 by 64, batch size as 96, drop out as 0.2 for LSTM Model. After 30 epochs training, our validation f1 score reached 0.83.

```
              precision    recall  f1-score   support

   drink_eat       0.88      0.73      0.80     13746
  phone_talk       0.72      0.89      0.80     13319
      selfie       0.91      0.79      0.85     11297
   sit_squat       0.80      0.82      0.81      6884
        walk       0.90      0.93      0.91     12817

    accuracy                           0.83     58063
   macro avg       0.84      0.83      0.83     58063
weighted avg       0.84      0.83      0.83     58063
```

#### Confusion Matrix

![Image of confusion matrix](https://github.com/fengyaoluo/Human-Activity-Monitoring-in-Public-Places/blob/main/images/Confusion%20Matrix.png)

### Demo

https://user-images.githubusercontent.com/59941969/128452369-641ba77b-cba6-448f-9d87-174bfaf6d7ca.mp4


### Run

#### Docker image: seancampos/final_proj_trt_pose

```
docker run --runtime nvidia -it --rm --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device /dev/video0 -v /YOUR_DATA_DIR:/data seancampos/final_proj_trt_pose
```

##### To build your own image
```
cd docker
docker build -t final_proj_trt_pose .
```


#### Train Body Pose Model

```
cd trt_pose
source ../tasks/human_pose/download_coco.sh
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
python3 preprocess_coco_person.py annotations/person_keypoints_train2017.json annotations/person_keypoints_train2017_modified.json
python3 preprocess_coco_person.py annotations/person_keypoints_val2017.json annotations/person_keypoints_val2017_modified.json
train.py ../tasks/human_pose/experiments/resnet18_baseline_att_368x2368_A.json
```

#### Evaluate the Body Pose Model

Run `src/eval_body_pose.ipynb` notebook.
Edit the notebook to change the model checkpoint if necessary.

#### Quantize the Body Pose Model

Run the `src/convert.ipynb` notebook.  
Edit the notebook to change the model if necessary.


#### Label Body Pose Vector on NTU RGB+D Videos

Put videos you wish to label from the NTU RGB+D dataset into folders organized by activity class.
```
/data/ntu/sit
/data/ntu/walk
...
```
Run `src/label_poses.py`
Edit the script if you wish to change the model, activity classes or LSTM time window.

#### Train the LSTM

Run `src/Train_LSTM.ipynb`
Edit the script if you wish to change the model, activity classes or LSTM time window.

#### Run inference on a video stream or file

To annotate a video:
```
python3 src/inference.py video.mp4
```
or to annotate a webcam stream in real-time:
```
python3 src/inference.py 0
```

### Or use our pretrained models

https://drive.google.com/drive/folders/1d3QT6lE4xpZWdylZvfP-jJeHw6XfF0To?usp=sharing


