#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

This project was done for Udacity's Self Driving Car Nanodegree program. The model performance was tested on 160x320x3.  It is a supervised regression problem between 
the car steering angles and the road images in front of a car.   

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
###Files Submitted 

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run3.mp4 - video file for track 1

model.py uses python generator with batch size (32 to 256 depending on the computing sources i used). First, I used AWS GPU instance. Later, I used google cloud instance with GPU.
Occasionaly, I used my laptop.

###Model Architecture and Training Strategy

The final design is based on the architecture design which is used by NVIDIA for the end-to-end self driving test. 
(https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
The design plot can be found :
./examples/nVidia_model.png

- I used Lambda layer to normalize input images .
- I've added dropout layers to avoid overfitting and increase dropout rates. I used only track 1 for addtional data and dropout is needed to avoid overfitting.
- I've also included RELU for activation function for convolutional layers
- Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE). 
- Using checkpoint, we saved the best model(kera's default is last one).

Model Architecture :

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Dropout    : 0.1
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Dropout    : 0.2
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Dropout    : 0.2
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Dropout    : 0.2 
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Dropout    : 0.3
- Fully connected: neurons: 100
- Dropout                 : 0.4
- Fully connected: neurons:  50
- Dropout                 : 0.4
- Fully connected: neurons:  10
- Dropout                 : 0.4
- Fully connected: neurons:   1 (output)

Model.summary():
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 158, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 77, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 37, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 3, 35, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       dropout_4[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 1, 33, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 2112)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      dropout_6[0][0]                  
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_7[0][0]                  
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_8[0][0]                  
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_9[0][0]                  
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0


#### Reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Overall, the convolutional models are less 
likely to overfit. When it overfits, dropout needs to be used in multiple places. In our case, we are using
data collected on track 1 as training and validation. Then, we are testing our model on track 1 simulator.
So, we need to use dropout. 

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I tried different inital values
and it did not affect much on the final result because i used enough number of epochs. 

## Data Preprocessing

Training data was chosen to keep the vehicle driving on the road. I collected additional data using track 1.
The additional data with angle = 0 has been removed. The total data with non-zero steering angle is around
half of total data. The angle distribution is relatively balanced and the image can be found in examples.

./examples/dist_stearing_angle.jpg 

We randomly distored training sample by increasing and decreasing brightness, and shifting vertically.
Images from center camera and the left and right side cameras with steering angle correction are used.
Also, all images are flipped. examples pictures can found in 

/examples/*jpg

### Image Sizing

- the images are cropped. (70 from top, 25 from bottom )
- the images are resized to 75x200 (3 color chanels)
- the images are normalized (image data divided by 255 and subtracted 0.5) 

### Image Augumentation

For training, I used the following augumentation techniques along with Python generator:

- For left image, steering angle is adjusted by +0.25
- For right image, steering angle is adjusted by -0.25
- Flip images with larger angles(angle > 0.35) 
- Random altering image brightness (lighter or darker)
- Random shadow 

### Training, Validation and Test

I splitted datasets into train and validation set and ratio is 0.2. Testing was done using the simulator.
I felt that the validation loss is not a great indication of how well it drives for choosing models.
But, it helped me to tune parameters. 

As for training, I used
- mean squared error for the loss function.
- Adam optimizer for the default initial rate(0.001).

### Track 1

I set the samples per epoch to number of traing datasets. I tried from 1 to 100 epochs but I found 10 epochs 
is good enough to produce a well trained model for track 1. I set number of epochs to 50 for most tests.
I did not collect data on track 2 and did not use track 2 for training. 
The batch size of 256 was chosen for google cloud GPU instance. When I use my own laptop, i chose 32 for the batchsize.

I enjoyed this project thoroughly and pleased with the result. However, there are many ways to improve the model. 
If i had a joystick, i could have collected much better training datasets and could have collected data on track 2.
Unfortunately, i am not good at playing video game and collected barely enough addtional training datasets for track 1 using keyboard.
Also, I could have normalized the datasets based on steering angle and other qualities. 





