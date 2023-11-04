# IMPORT PACKAGES ------------------------------------------------------------------------------------------------
import os
import cv2 #pip install opencv-python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Import tensorflow packages 
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
plt.imshow(X_train[2], cmap='gray')  # Display the first training image
plt.title(f"Label: {y_train[2]}")  # Display the label corresponding to the image
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
model.add(Dropout(0.2))

# hidden layers
# dense layer is most basic NN layer where each neuron is connected to each other neuron of the other layer
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))

#model.add(Dense(units=32, activation='relu'))
#model.add(Dropout(0.2))

# 2 dense hidden layers with 128 units had loss: 0.07195307314395905 and accuracy: 0.9797999858856201
# 2 dense hidden layers with 128 units, then 64 had loss: 0.07154695689678192 and accuracy: 0.980400025844574
# 3 dense hidden layers with 128, 64, 32 had loss: 0.07737532258033752 and accuracy: 0.980400025844574

#output layer
# softmax makes a probability distribution so all add to 1 and biggest % you choose is the predicted class
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(x=X_train,
          y=y_train,
          batch_size=256,
          epochs=100,
          validation_data=(X_test,y_test), 
          verbose=1,
          callbacks=[early_stop]
         )

# SAVE MODEL -----------------------------------------------------------------------------------------------------
# model.save("handwritten_digits.model")

# !!! Load model - saves all the time from reraining again
# saved_model = load_model("handwritten_digits.model")

# EVALUATE MODEL ON TEST DATA ------------------------------------------------------------------------------------
# 1) plot training vs test losses 
losses = pd.DataFrame(model.history.history)[["loss", "val_loss"]]
losses.plot()
plt.show()

# 2) get confusion matrix and classification report 
from sklearn.metrics import classification_report, confusion_matrix
predictions = np.argmax(model.predict(X_test), axis =1) #find the max in each row
predictions

print(classification_report(y_test, predictions))
print("\n")
print(confusion_matrix(y_test,predictions))

# 3) get loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)

print(loss) # want low
print(accuracy) # want high to 1 as possible

# EVALUATE MODEL ON OWN DRAWN DATA -------------------------------------------------------------------------------
# loading and processing a single image
trial = cv2.imread(f"digits/10.png")[:,:,0] #load image in grayscale mode - shape (28,28)
trial = np.invert(np.array([trial])) #inverts pixel value of the image array
# the np.array([]) around trial changes the shape to (1,28,28) so we have 1 sample of the image shape (28,28) to run our model on
trial = normalize(trial, axis=1)
trial.shape
prediction = model.predict(trial)
plt.imshow(trial[0], cmap="gray") #displays image on grayscale
plt.title(f"Label: {np.argmax(prediction)}") # np.argmax returns the indices of the maximum values along an axis
plt.show()

# looping through multiple images
predicted_data=[]
image_number = 0
while os.path.isfile(f"digits/{image_number}.png"):
    img = cv2.imread(f"digits/{image_number}.png")[:,:,0]
    img = np.invert(np.array([img]))
    img = normalize(img, axis = 1)
    prediction = model.predict(img)
    
    # show graph of true value with predicted result
    plt.imshow(img[0], cmap="gray") #can also make cmap=plt.cm.binary
    plt.title(f"Label: {np.argmax(prediction)}")
    plt.show()
    
    # get list of predicted value
    predicted_data.append(np.argmax(prediction))
    
    image_number += 1

predicted_data
true_data = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]

acc = 0
for i in range(20):
    if predicted_data[i] == true_data[i]:
        acc +=1

print(f"Accuracy of drawn results is: {acc/20}")
