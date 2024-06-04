import numpy as np
import os
import cv2
import pickle
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_path = 'FER-2013'
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


def preprocess(mode='train'):
    try:
        images = np.load(f'images_{mode}.npy')
        labels = np.load(f'labels_{mode}.npy')
        print("Loaded Preprocessed files")

    except FileNotFoundError:
        images = []
        labels = []
        for emotion in emotion_labels:
            image_files = glob(os.path.join(data_path, mode, emotion, '*.png'))
            for file in image_files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(emotion_labels.index(emotion))

        images = np.array(images)
        labels = np.array(labels)

        images = images / 255.0
        images = np.expand_dims(images, axis=-1)

        np.save(f'images_{mode}.npy', images)
        np.save(f'labels_{mode}.npy', labels)

        print("Preprocessing Done")

    return images, labels


def train_model(X_train, Y_train, X_test, Y_test):
    try:
        model = load_model('emotion_detection_model.h5')
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        print("Loaded Saved Model")

    except (OSError, FileNotFoundError):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(emotion_labels), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(X_train)

        history = model.fit(datagen.flow(X_train, Y_train, batch_size=32), epochs=25, validation_data=(X_test, Y_test))

        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        history = history.history
        model.save('emotion_detection_model.h5')

        print("Model Trained")

    return model, history


# Function to evaluate the model
def evaluate(X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')


# Function to plot training history
def plot_history(history):
    plt.plot(history['accuracy'], label='Accuracy (training)')
    plt.plot(history['val_accuracy'], label='Accuracy (validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, X_test, Y_test, labels):
    predictions = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(Y_test, predictions, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.show()


X_train, Y_train = preprocess(mode='train')
X_test, Y_test = preprocess(mode='test')
model, history = train_model(X_train, Y_train, X_test, Y_test)
evaluate(X_test, Y_test)
plot_history(history)
plot_confusion_matrix(model, X_test, Y_test, emotion_labels)

