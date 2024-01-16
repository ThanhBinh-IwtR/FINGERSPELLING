<h1 align="center"> Sign Language </h1>

![Demo](assets/demo.png)
## 👋 Introduction

In this project, We will create a model trained on largest dataset of its kind, release specially for [Google - American Sign Language Fingerspelling Recognition Competition](https://www.kaggle.com/competitions/asl-fingerspelling). **Tensorflow** are framework that mainly using in this project. 

One thing you need to know that, We make this project **just for learning**, so that this model performance can not be better than another SOTA models.

We also deploy it in local host so that it limited of client using and and we will not discuss it more here. 

## 👤 Contributors
**I would like to express my sincere thanks to the these wonderful people who have contributed to this project with me:**

- Hoa D. (Leader)
- Tung N. (contributor)
- Binh P. (contributor)
- Hai V. (contributor)
- Lam N. (contributor)
- Duc P. (contributor)

## 📝 Work Flow
![Alt text](assets/workflow.png)
### A. Data
1. #### Raw Data | [Download](https://www.kaggle.com/competitions/asl-fingerspelling/data)
`path` - The path to the landmark file.

`file_id` - A unique identifier for the data file.

`participant_id` - A unique identifier for the data contributor.

`sequence_id` - A unique identifier for the landmark sequence. Each data file may contain many sequences.

`phrase` - The labels for the landmark sequence. The train and test datasets contain randomly generated addresses, phone numbers, and urls derived from components of real addresses/phone numbers/urls.

2. #### EDA | [CODE](https://github.com/ThanhBinh-IwtR/SIGN_LANGUAGE/blob/main/data/asl-eda.ipynb)

There are 1630 landmarks in orginal data, but we just use 21 landmarks from each hand and 80 landmarks from lip in this project.

![This image illustrates the 21 keypoints on the hand](assets/hand_landmarks.png)
*This image illustrates the 21 landmarks on the hand*

source: [MediaPipe](https://google.github.io/mediapipe/solutions/hands.html)

![lip landmarks](assets/lip_landmarks.png)
*This image illustrates the landmarks on the lip* 

source: [MediaPipe](https://google.github.io/mediapipe/solutions/face_mesh.html)

This is just demo a part of EDA, please check full code.

3. #### Preprocessing - CODE UPDATE LATER

### B. Model | [CODE](https://github.com/ThanhBinh-IwtR/SIGN_LANGUAGE/tree/main/Project)
1. #### Model Architecture
This image below is showing on horizontal view for better experience of getting general knowledge, but you can see the vertical version via this [link](assets/model.tflite.png)

![Model Architecture](assets/model.tflite_horizon.png)

2. #### Training
3. #### Evaluation
5. #### Model Performance

### C. Web | [CODE](https://github.com/ThanhBinh-IwtR/SIGN_LANGUAGE/tree/main/Web)
### D. Future Work
This project is serve for our learning purpose, it just have limited time so that we will transfer to Vietnamese in the future. All of our update will be in this repository.
### E. Paper:
We will release our paper in the future.