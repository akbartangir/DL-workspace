import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import random
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# === CONFIG ===
dataset_path = "/Users/box/Desktop/python/DL-workspace/MNIST-CNN/Train"
label_file = os.path.join(dataset_path, "labels.csv")
batch_size_val = 32
epochs_val = 10
image_dimensions = (32, 32, 3)
test_ratio = 0.2
validation_ratio = 0.2

# === LOAD IMAGES ===
images = []
classNo = []

myList = os.listdir(dataset_path)
print("Total Classes Detected:", len(myList))
no_of_classes = len(myList)
print("Importing Classes.....")

for class_index in range(no_of_classes):
    class_folder = os.path.join(dataset_path, str(class_index))

    if os.path.isdir(class_folder):
        image_files = os.listdir(class_folder)
        for img_file in image_files:
            img_path = os.path.join(class_folder, img_file)
            cur_img = cv2.imread(img_path)

            if cur_img is not None:
                try:
                    cur_img = cv2.resize(cur_img, (image_dimensions[0], image_dimensions[1]))
                    images.append(cur_img)
                    classNo.append(class_index)
                except Exception as e:
                    print(f"Error resizing image {img_path}: {e}")
            else:
                print(f"Could not read image {img_path}")
        print(class_index, end=" ")
    else:
        print(f"Folder {class_index} does not exist.")

print("\nImage loading complete.")
images = np.array(images)
classNo = np.array(classNo)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)

print("Data Shapes")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

# === PREPROCESSING ===
def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img): return cv2.equalizeHist(img)
def preprocessing(img): return equalize(grayscale(img)) / 255.0

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Add channel dimension
X_train = X_train.reshape(X_train.shape[0], image_dimensions[0], image_dimensions[1], 1)
X_validation = X_validation.reshape(X_validation.shape[0], image_dimensions[0], image_dimensions[1], 1)
X_test = X_test.reshape(X_test.shape[0], image_dimensions[0], image_dimensions[1], 1)

# === DATA AUGMENTATION ===
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# === ONE-HOT ENCODING ===
y_train = to_categorical(y_train, no_of_classes)
y_validation = to_categorical(y_validation, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)

# === MODEL ===
def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(image_dimensions[0], image_dimensions[1], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
model.summary()

# === TRAINING ===
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=len(X_train) // batch_size_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=True)

# === PLOTTING ===
plt.figure(1)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')

plt.show()

# === EVALUATION ===
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# === SAVE MODEL ===
model.save("model2.h5")
