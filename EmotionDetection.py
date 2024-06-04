import numpy as np
import os
import cv2
from glob import glob
from sklearn.model_selection import train_test_split


data_path = 'FER-2013'
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


def preprocess(mode = 'train'):
    try:
        images = np.load('images.npy')
        labels = np.load('labels.npy')
        print("Loaded Preprocessed files")

    except FileNotFoundError:
        images = []
        labels = []
        for emotion in emotion_labels:
            image_files = glob(os.path.join(data_path, mode, emotion, '*.jpg'))
            for file in image_files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(emotion_labels.index(emotion))

        images = np.array(images)
        labels = np.array(labels)

        images = images / 255.0
        images = np.expand_dims(images, axis=-1)

        np.save('images.npy', images)
        np.save('labels.npy', labels)

        print("Preprocessing Done")

    return train_test_split(images, labels, test_size=0.2)