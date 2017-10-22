import pandas as pd
import numpy as np
import csv
#import cv2
import keras as kr
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#import cv2
import numpy as np
import sklearn

"""def pre_process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    image[:,:,0] = clahe.apply(image[:,:,0])
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    return image
"""
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                #img_center = cv2.imread(name)
                image = Image.open(name)
                image.load()
                img_center = np.asarray(image)
                #img_center = pre_process_image(img_center)
                name = './data/IMG/'+batch_sample[1].split('/')[-1]
                #img_left = cv2.imread(name)
                #img_left = pre_process_image(img_left)
                image = Image.open(name)
                image.load()
                img_left = np.asarray(image)
                name = './data/IMG/'+batch_sample[2].split('/')[-1]
                #img_right = cv2.imread(name)
                #img_right = pre_process_image(img_right)
                image = Image.open(name)
                image.load()
                img_right = np.asarray(image)
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.25 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/225.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(16, 8, 8, subsample=(2,2), border_mode="same"))
model.add(Convolution2D(16, 8, 8, subsample=(2,2), border_mode="same"))
model.add(Convolution2D(16, 8, 8, subsample=(2,2), border_mode="same"))
model.add(ELU())
model.add(MaxPooling2D())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(MaxPooling2D())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1000))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(500))
model.add(ELU())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


history_object = model.fit_generator(train_generator, samples_per_epoch= 3*len(train_samples), validation_data=validation_generator, nb_val_samples=3*len(validation_samples), nb_epoch=5)
model.save('modelComma.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
