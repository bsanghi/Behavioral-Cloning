import csv
import cv2
import numpy as np
import os
import argparse

import sklearn
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# input variables
EPOCHS = 50
BATCH_SIZE = 32
filename = 'crl_dropout_035anglesflipped_02'

# loading data
def load_data():
    '''
    Load training data and split it into training and validation set
    '''
    samples_t = []
    with open('data/total_nonzero_driving_log_noheader.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
        	samples_t.append(line)

    samples_t = sklearn.utils.shuffle(samples_t, random_state=41)
    train_samples, validation_samples = train_test_split(samples_t, test_size=0.2)

    return train_samples, validation_samples

# distort 
def random_distort(img, angle):
    ''' 
    method for adding random distortion to dataset images, including random brightness adjust and shadow
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor

    return (new_img.astype(np.uint8), angle)

# generator
def generator(samples, batch_size=BATCH_SIZE, validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images, then yield
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples=sklearn.utils.shuffle(samples, random_state=41)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    correction = 0.0
                    if i == 1:
                        correction = 0.25
                    elif i == 2:
                        correction = -0.25

                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])+correction
                    images.append(image)
                    angles.append(angle)
                    # distort
                    if not validation_flag:
                        img,angle = random_distort(image,angle)
                        images.append(img)
                        angles.append(angle)

                    # flipping
                    if abs(angle) > 0.35:
                    	images.append(cv2.flip(image,1))
                    	angles.append(-1.0*angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def build_model():
    '''
    Modified NVIDIA model
    '''
    
    model= Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    #model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(Dense(50))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.summary()

    return model

def train_model(model, train_generator, validation_generator, samples_per_epoch, nb_val_samples, model_filename):
    '''
    Train the model
    '''
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    
    Adam = optimizers.Adam(lr=0.001)

    model.compile(loss='mse', optimizer=Adam)

    history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, \
                               nb_epoch=EPOCHS, validation_data=validation_generator, \
                               nb_val_samples=nb_val_samples,callbacks=[checkpoint])
    model.save(model_filename)
    return history_object

def plot_loss(history_object, loss_filename):
    """
    Plot mse loss for training and validation datasets 
    """
    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(loss_filename)

def main():
    """
    Load train/validation data set and train the model
    """

    # load data
    train_samples, validation_samples = load_data()
    # compile and train the model using the generator function
    train_generator = generator(train_samples, validation_flag=False)
    validation_generator = generator(validation_samples, validation_flag=True)
    # build model
    model = build_model()

    # train model
    nb_val_samples=len(validation_samples)
    samples_per_epoch = int(len(train_samples) / BATCH_SIZE) * BATCH_SIZE
    model_filename='model_epoch'+str(EPOCHS)+'_'+filename+'.h5'

    history_object=train_model(model,train_generator,validation_generator, samples_per_epoch,nb_val_samples, model_filename)

    # plot loss plot
    loss_filename='loss_epoch'+str(EPOCHS)+'_'+filename+'.jpg'
    plot_loss(history_object, loss_filename )

if __name__ == '__main__':
    main()

