import pandas as pd
import numpy as np
import csv
import cv2
import keras as kr
import matplotlib as mb

def readData():
    file1 = pd.read_csv(filepath_or_buffer="./data/driving_log.csv", header=None, names=["center", "left", "right", "steering", "throttle", "brake",'speed'])
    file2 = pd.read_csv(filepath_or_buffer="./data/driving_log1.csv", header=None, names=['center', 'left', 'right', 'steering', 'throttle', 'brake','speed'])

    file1['center'] = "./data/" + file1['center'].str.strip()
    file1['left'] = "./data/" + file1['left'].str.strip()
    file1['right'] = "./data/" + file1['right'].str.strip() 


    file2['center'] = file2['center'].str.replace("/home/vipul", "./data")                  
    file2['left'] = file2['left'].str.replace("/home/vipul", "./data") 
    file2['right'] = file2['right'].str.replace("/home/vipul", "./data") 

    data = pd.concat([file1, file2])
    #Speed 0 is not driving representation
    data = data[data['speed'] != 0].copy(deep = True)
    #Data augmentation to be done
    dataReverse = data.copy(deep=True)
    dataReverse['steering'] = -1.0*dataReverse['steering']
    images = []
    paths = data['center'].tolist()
    for path in paths:
        image = cv2.imread(path)
        images.append(image)

    X_train = np.array(images)
    y_train = data['steering'].as_matrix()

    images = []                                                                
    paths = dataReverse['center'].tolist()                                            
    for path in paths:                                                         
        image = cv2.imread(path)
        image_flipped = np.fliplr(image)
        images.append(image)

    X_trainR = np.array(images) 
    y_trainR = dataReverse['steering'].as_matrix() 
    
    X_train = np.concatenate((X_train, X_trainR))
    y_train = np.concatenate((y_train, y_trainR))
    print(X_train.shape)
    print(y_train.shape)
    return X_train,y_train



if __name__ == "__main__":

    X_train, y_train = readData()
    
    from keras.models import Sequential

    from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda

    from keras.layers.convolutional import Convolution2D

    from keras.layers.pooling import MaxPooling2D


    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
    model.add(Lambda(lambda x:x / 255.0 - 0.5))
    #model.add(Flatten())
    #model.add(Dense(1))
    model.add(Convolution2D(6, 5, 5, activation="relu"))

    model.add(MaxPooling2D())

    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Convolution2D(6, 5, 5, activation="relu"))                       
                                                                                    
    model.add(MaxPooling2D())
    
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())

    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(Dense(84))

    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, batch_size=16)
    model.save('modelLENET.h5')


