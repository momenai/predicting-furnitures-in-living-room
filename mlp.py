from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from scipy.misc import imread, imsave, imresize
import os
import glob
from os import listdir
from os.path import isfile, join
import random
import numpy as np
from scipy import io
import cv2
from scipy.io import loadmat

#the directory which is our dataset
DIR = 'train'

#take the number of images
x = len(glob.glob('train\*'))
print(x)


#specify the labels of the each classes according the name of the image and return and 1D array whose length is 10.
#it takes one parameter which is the name of the image
def label_img(name):
    word_label = name.split('-')[0]
    if word_label == 'armchair': return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif word_label == 'carpet' : return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif word_label == 'lamp' : return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif word_label == 'painting' : return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif word_label == 'sofa' : return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif word_label == 'sofa_table' : return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif word_label == 'television' : return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif word_label == 'television_unit' : return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif word_label == 'vases' : return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif word_label == 'coffe_table' : return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

# datasets are created. 
# Each image converted to [768x1] matrixes
# After datasets are shuffled they are printed the train.mat and test.mat files
def load_training_data():
    train_data = []
    img_arr = []
    img_label =[]
    for img in os.listdir(DIR):
        label = label_img(img)
        path = os.path.join(DIR, img)
        if "DS_Store" not in path:
			#convert images to [768x1] matrices
            img = imread(path)
            img = imresize(img,[32,24])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.flatten()
            img_arr.append(img)
            img_label.append(label)
            
            
    c = list(zip(img_arr, img_label))

    random.shuffle(c)
	#split the dataset to the train and test and write them to the files
    img_arr, img_label = zip(*c)
    data={'x':img_arr[700:], 'y':img_label[700:]}
    io.savemat('train.mat',data)
    data={'x':img_arr[0:700], 'y':img_label[0:700]}
    io.savemat('test.mat',data)

load_training_data()



#train.mat is read 
train = loadmat("train.mat")
#take the array of each images
train_x = np.array(train["x"]) 
X_train = (train_x - np.mean(train_x)) / (10.0 * np.std(train_x))
#take the label of each images
y_train = train["y"]

#test.mat is read
test = loadmat("test.mat")
#take the array of each images
test_x = np.array(test["x"])
X_test = (test_x - np.mean(test_x)) / (10.0 * np.std(test_x))
#take the label of each images
y_test = test["y"]

print(len(X_train))
print(len(y_train))
print(y_train)

# network is set up
model = Sequential()
model.add(Dense(522, input_dim=768, activation='relu'))
model.add(Dense(351, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#network is training
model.fit(X_train, y_train, batch_size = 10, epochs = 5)

# test results
y_pred = model.evaluate(X_test,y_test, verbose=1,steps=20)
#accuracy
print(y_pred[1])

