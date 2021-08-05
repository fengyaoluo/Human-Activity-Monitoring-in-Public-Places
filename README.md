# Human Activity Monitoring in Public Places
### Sean Campos, Praveen Reddy Kasireddy, Sam Shih, Fengyao Luo
Introduing a data pipeline with Deep Learning models to recogining human actions in public places 

Research Paper

Presentation

### Goal
- Build an end to end data pipeline to recognize human actions in the public places
- Output an aggregated datapoint to a live time dashboard

### Abstract



This project aims at collecting metrics pertaining to the activities of a person or a group of people based on sensor observation in the public realm, in order to inform better planning decisions. The application of activity recognition can be used in calculating the time span for human activities in public spaces such as parks. By leveraging the data, stakeholders can enhance the environment as well as the utilization of public facilities including benches, playgrounds, food areas, etc, as well as plan for maintenance, capital improvements and events.

In this project, we use an edge device, Jetson with a web camera to shoot a video, frame a person in the video, identify body parts, and recognize human postures. The pipeline is constructed with 2 main parts: Body Pose Detector and Action Recognition. The body pose detector used the Resnet18 as our backbone and trained it with the COCO dataset. It is able to pinpoint the coordinates of 18 possible body parts and draw connections to each, forming a skeleton of 2D human pose. The action recognition used an LSTM to track a sequence of frames with pose vectors and trained with NTU RGB+D 120 dataset. The dataset includes 120 categories and we only picked 5 categories that were germain to our problem domain to train our model. The Identified activities include drink and eat, sit and squat, phone and talk, walk, selfie. We achieved a f1 score of 0.83 on the test dataset. Each category’s accuracy score has reached above 0.80 in the end.

In the ideal case, an activity is recognized regardless of the environment it is performed in or the performing person and data can be processed at a near real time pace. Instead of connecting a real time camera, we shoot a video which includes the actions to test our models, and output the measurable metrics, including actions, number of people, time span.

### Data Pipeline

#### 

**Camera:**
- Stream video to the inference engine on the Edge Device


**Edge Device(Jetson):**
- Extract information about people and their body parts
- Perform  object tracking
- Detect activity
- Record the metrics


#### Phase 1: Body Part Detection

#### Phase 2: Action Recognition Detection

### Evaluation Results

### Demo

### Run

