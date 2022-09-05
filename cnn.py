# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import plot_model
from keras.layers import Dropout
from keras.layers. normalization import BatchNormalization
# Initialising the CNN
from keras.metrics import categorical_accuracy
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#define the model
classifier = Sequential()

## architure or model.

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))


classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 384, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['categorical_accuracy'])
classifier.summary()

#image processing skills
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#datasets are laoded
training_set = train_datagen.flow_from_directory('../../dataset/train/',target_size = (64, 64),batch_size = 128,class_mode = 'categorical')
val_set = train_datagen.flow_from_directory('../../dataset/val/',target_size = (64, 64),batch_size = 128,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('../../dataset/test_set/',target_size = (64, 64),batch_size = 128,class_mode = 'categorical')
#network is trained
history= classifier.fit_generator(training_set,steps_per_epoch = 25,epochs = 100,validation_data = val_set,validation_steps = 100, shuffle=False)

#test accuracy is calculated
scores = classifier.evaluate_generator(test_set,700) #700 testing images
print("Accuracy = ", scores[1])
#print("Loss = ", scores[0])

#accuracy graph is plotted
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

#loss graph is plotted
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')