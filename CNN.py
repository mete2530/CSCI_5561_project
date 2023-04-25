import os
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as keras
import random
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from keras.models import load_model


def data():
    folder = 'CSCI 5561 Resized Images'
    trainFiles = []
    testFiles = []

    fileLabel = {}
    classNum = 0
    for classDir in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder,classDir)):
            continue
        classLabel = classNum
        classFiles = os.listdir(os.path.join(folder,classDir))
        random.shuffle(classFiles)
        numTrain = int(len(classFiles) * 0.8)
        for i,fileName in enumerate(classFiles):
            if not fileName.endswith('.png'):
                continue
            filePath = os.path.join(classDir,fileName)
            fileLabel[filePath] = classLabel
            if i < numTrain:
                trainFiles.append((filePath,classLabel))
            else:
                testFiles.append((filePath,classLabel))
        classNum += 1

    return trainFiles, testFiles, fileLabel


def dataGenerator(trainData, testData, labels, batch_size=32, img_size=(256, 256)):
    # train_filepaths = [os.path.join('CSCI 5561 Resized Images', filepath) for filepath, label in trainData]
    # train_labels = [label for filepath, label in trainData]

    # test_filepaths = [os.path.join('CSCI 5561 Resized Images', filepath) for filepath, label in testData]
    # test_labels = [label for filepath, label in testData]

    trainGen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    trainGenerator = trainGen.flow_from_directory(
        'CSCI 5561 Resized Images',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    
    testGen = ImageDataGenerator(rescale=1./255)
    testGenerator = testGen.flow_from_directory(
        'CSCI 5561 Resized Images',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
    
    return trainGenerator, testGenerator


def trainModel(trainDataGenerator):
    # VGG19 model architecture
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=trainDataGenerator.target_size + (3,)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(trainDataGenerator.num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainDataGenerator, steps_per_epoch=trainDataGenerator.samples//trainDataGenerator.batch_size, epochs=10)
    model.save('xray_model_weights.h5')


def testModel(test_data, test_labels):
    model = load_model('xray_model_weights.h5')

    predictions = []
    true_labels = []
    for filepath in test_data:
        img = cv2.imread(os.path.join('CSCI 5561 Resized Images', filepath[0]))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)[0]
        predictions.append(np.argmax(pred))
        
    for i in range(len(test_data)):
        true_labels = np.array(test_data[i][1])
    accuracy = np.mean(predictions == true_labels)
    print(f'Test Accuracy: {accuracy:.2%}')
    
    return predictions
   
 
Train,Test,Labels = data()
trainGen, testGen = dataGenerator(Train,Test,Labels)
trainModel(trainGen)
prediction = testModel(Test,Labels)


