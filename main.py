# IMPORT PACKAGES ------------------------------------------------------------------------------------------------
import os
import cv2 #pip install opencv-python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Import tensorflow packages 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# DOWNLOAD DATASET -----------------------------------------------------------------------------------------------
# Load the data into training and testing sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Explore the shape of the training and testing data
X_train.shape  # (60000, 28, 28) - 60,000 images of size 28x28 pixels
y_train.shape  # (60000,) - 60,000 labels (digits 0 to 9)
X_test.shape  # (10000, 28, 28) - 10,000 images of size 28x28 pixels
y_test.shape  # (10000,) - 10,000 labels (digits 0 to 9)

# Visualize a sample image from the dataset
# plt.imshow() used to visualise images in 2D arrays
plt.imshow(X_train[0], cmap='gray')  # Display the first training image
plt.title("Label: " + str(y_train[0]))  # Display the label corresponding to the image
plt.show()

# Check distribution of the y set
pd.DataFrame(y_train).value_counts()
sns.histplot(y_train)
plt.show()
# looks like a balanced dataset

# NORMALIZE THE DATASET -----------------------------------------------------------------------------------------
# Make it so every value is between 0 and 1 - normalize the pixels so it's easier for neural network
X_train = normalize(X_train, axis = 1)
X_test = normalize(X_test, axis = 1)

# CREATE NEURAL NETWORK -----------------------------------------------------------------------------------------
model = Sequential()

# input layer
# flatten layer changes our grid from 28 x 28 to 1 line of 784 pixels
model.add(Flatten(input_shape=(28,28)))

# hidden layers
# dense layer is most basic NN layer where each neuron is connected to each other neuron of the other layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))

#output layer
# softmax makes a probability distribution so all add to 1 and biggest % you choose is the predicted class
model.add(Dense(units=10, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
                optimizer ='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)

# SAVE MODEL -----------------------------------------------------------------------------------------------------
model.save("handwritten_digits.model")

# !!! Load model - saves all the time from reraining again
# saved_model = load_model("handwritten_digits.model")

# EVALUATE MODEL ON TEST DATA ------------------------------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print(loss) # want low
print(accuracy) # want high to 1 as possible

# EVALUATE MODEL ON OWN DRAWN DATA -------------------------------------------------------------------------------
predicted_data=[]
image_number = 0
while os.path.isfile(f"digits/{image_number}.png"):
    img = cv2.imread(f"digits/{image_number}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"This digit is probably a {np.argmax(prediction)}")
    predicted_data.append(np.argmax(prediction))
    plt.imshow(img[0], cmap=plt.cm.binary) #can also make cmap='gray'
    #plt.show()
    image_number += 1

predicted_data
true_data = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]

acc = 0
for i in range(20):
    if predicted_data[i] == true_data[i]:
        acc +=1

print(f"Accuracy of drawn results is: {acc/20}")