import os
import csv
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import sklearn
import matplotlib.image as mpimg
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense , Flatten, Lambda , Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    skip = True
    for line in reader:
        #skip the headings
        if skip:
            skip = False
            continue
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#getting the count of images which will be augmented
#using this step to get the exact count of samples_per_epoch in model.fit_generator
cnt = 0
for i in train_samples:
    if(float(i[3])!=0.0):
        cnt +=3
print(cnt)





IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

def preprocess(image):
     #crop the top 60 and bottom 25 pixels
    image = image[60:-25, :, :]
    #resize image
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    #covert to yuv nvidia model uses yuv input
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    #Normalize image
    image = image.astype(np.float32)
    image = image/255.0 - 0.5
    return image

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name)
                #pre process the image
                center_image = preprocess(center_image)
                #get the steering angle
                center_angle = float(batch_sample[3])
                #append the image and angle to a list
                images.append(center_image)
                angles.append(center_angle)
                
                if(center_angle != 0.0):
                    #add left image
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    #fetch image
                    left_image = mpimg.imread(name)
                    #pre process the image
                    left_image = preprocess(left_image)
                    #get the steering angle
                    left_angle = float(batch_sample[3])
                    #add a correction factor of 0.2
                    left_angle += 0.20
                    #append the image and angle to a list
                    images.append(left_image)
                    angles.append(left_angle)
                    
                    #add right image
                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    #fetch image
                    right_image = mpimg.imread(name)
                    #pre process the image
                    right_image = preprocess(right_image)
                    #get the steering angle
                    right_angle = float(batch_sample[3])
                    #add a correction factor of -0.2
                    right_angle -= 0.20
                    #append the image and angle to a list
                    images.append(right_image)
                    angles.append(right_angle)
                    
                    #add a center image flipped
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                    #fetch image
                    center_image = mpimg.imread(name)
                    #flip the image vertically
                    center_image = cv2.flip(center_image,1)
                    #pre process the image
                    center_image = preprocess(center_image)
                    #fetch the steering angle
                    center_angle = float(batch_sample[3])
                    #negate the angle
                    center_angle = -center_angle
                    #append the image and angle to a list
                    images.append(center_image)
                    angles.append(center_angle)
            #convert the images and steering angles in list to numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            #shuffle and yield
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

def getModel():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2),input_shape=(66,200,3)))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

model = getModel()

model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint("final.h5", verbose=1, save_best_only=True)
callbacks_list = [checkpointer]
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)+cnt, validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=10,callbacks=callbacks_list)
print('Done')