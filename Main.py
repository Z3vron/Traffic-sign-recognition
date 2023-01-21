import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score

# Setting variables
amount_of_diffrent_signs = 21
data = []
labels = []

# Get all images
for i in range(amount_of_diffrent_signs):
    path = os.path.join(os.getcwd(), 'Train_photos', str(i))
    images_to_train = os.listdir(path)
    for image_to_train in images_to_train:
        try:
            image = Image.open(path + '\\' + image_to_train)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Image for training purpose can't be loaded")

# Prepare data
data = np.array(data)
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=20)
y_train = to_categorical(y_train, amount_of_diffrent_signs)
y_test = to_categorical(y_test,amount_of_diffrent_signs)

# Print basic information
print("Number of training images: " + str(x_train.shape[0]))
#print("Number of testing images: " + str(x_test.shape[0]))
print("Size of images: " + str(data.shape[1]) + "x" + str(data.shape[2]))

# Building the model
model = Sequential()

# Adding  layers
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape= x_train.shape[1:]))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2))) 
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(amount_of_diffrent_signs, activation = 'softmax'))
model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Training the model
epochs = 15
model.fit(x_train, y_train, batch_size = 32, epochs=epochs)
model.save("CNN_model.h5")