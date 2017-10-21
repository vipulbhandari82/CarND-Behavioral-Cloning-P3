import pandas as pd
import numpy as np
import csv
import cv2
import keras as kr
import matplotlib as mb

def readData():
    file1 = pd.read_csv(filepath_or_buffer="./data/driving_log.csv", header=None, names=["center", "left", "right", "steering", "throttle", "brake",'speed'])
    #file2 = pd.read_csv(filepath_or_buffer="./data/driving_log1.csv", header=None, names=['center', 'left', 'right', 'steering', 'throttle', 'brake','speed'])

    file1['center'] = "./data/" + file1['center'].str.strip()
    file1['left'] = "./data/" + file1['left'].str.strip()
    file1['right'] = "./data/" + file1['right'].str.strip() 


    #file2['center'] = file2['center'].str.replace("/home/vipul", "./data")                  
    #file2['left'] = file2['left'].str.replace("/home/vipul", "./data") 
    #file2['right'] = file2['right'].str.replace("/home/vipul", "./data") 
    
    data = file1
    #data = pd.concat([file1, file2])
    #Speed 0 is not driving representation
    data = data[data['speed'] != 0].copy(deep = True)
    #Data augmentation to be done
    #dataReverse = data.copy(deep=True)
    #dataReverse = dataReverse[~dataReverse['steering'].between(-5.0,5.0)].copy(deep=True)
    #dataReverse['steering'] = -1.0*dataReverse['steering']
    images = []
    paths = data['center'].tolist()
    for path in paths:
        image = cv2.imread(path)
        images.append(image)

    X_train = np.array(images)
    y_train = data['steering'].as_matrix()
    """
    images = []                                                                
    paths = dataReverse['center'].tolist()                                            
    for path in paths:                                                         
        image = cv2.imread(path)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)

    X_trainR = np.array(images) 
    y_trainR = dataReverse['steering'].as_matrix() 
    print(X_train.shape)
    print(X_trainR.shape) 
    X_train = np.concatenate((X_train, X_trainR))
    y_train = np.concatenate((y_train, y_trainR))
    """
    print(X_train.shape)
    print(y_train.shape)
    return X_train,y_train



if __name__ == "__main__":

    X_train, y_train = readData()
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
    from keras.layers.convolutional import Convolution2D
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0,
              input_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")


    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=2)
    model.save('modelComma.h5')


