import csv
import cv2
import matplotlib as plt
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


samples = []
with open('L1_L2_2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            correction = 0.2
            path = './TEST/'
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Center
                name = path+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                flipped_center_image = np.fliplr(center_image)
                flipped_center_angle = -center_angle
                images.append(flipped_center_image)
                angles.append(flipped_center_angle)
                # Left
                name = path+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = (float(batch_sample[3]))+correction
                images.append(left_image)
                angles.append(left_angle)
                flipped_left_image = np.fliplr(left_image)
                flipped_left_angle = -left_angle
                images.append(flipped_left_image)
                angles.append(flipped_left_angle)
                # Right
                name = path+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = (float(batch_sample[3]))-correction
                images.append(right_image)
                angles.append(right_angle)
                flipped_right_image = np.fliplr(right_image)
                flipped_right_angle = -right_angle
                images.append(flipped_right_image)
                angles.append(flipped_right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(36, (5,5), strides=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(48, (5,5), strides=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()


early_stopping_monitor = EarlyStopping(patience=2)

LEARNING_RATE = 1.0e-4

model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

# model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

# model.fit_generator(train_generator, samples_per_epoch= /
#            len(train_samples), validation_data=validation_generator, /
#           nb_val_samples=len(validation_samples), nb_epoch=3)

model.fit_generator(train_generator, samples_per_epoch = 200,
                    nb_epoch = 20,
                    validation_data = (validation_generator),
                    nb_val_samples = 10,
                    verbose = 1,
                    callbacks=[early_stopping_monitor])
# len(train_samples)
model.save('model3.h5')
print("Model Saved")
