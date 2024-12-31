import numpy as np
import os
import cv2
import pickle
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tkinter import Tk, filedialog
from sklearn.utils.class_weight import compute_class_weight

data_path = 'FER-2013'
emotion_labels = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


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
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4), input_shape=(48, 48, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(emotion_labels), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.2,
            horizontal_flip=True,
            shear_range=0.2
        )
        datagen.fit(X_train)
        # Add EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(Y_train),
            y=Y_train
        )
        class_weights_dict = dict(enumerate(class_weights))

        history = model.fit(
            datagen.flow(X_train, Y_train, batch_size=32),
            validation_data=(X_test, Y_test),
            epochs=50,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights_dict
        )

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


def predict_emotion(model, labels):
    # Open file dialog to select image
    Tk().withdraw()  # Hide the root Tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image File",
                                           filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

    if not file_path:
        print("No file selected.")
        return

    # Load and preprocess the image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (48, 48)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))  # Add batch and channel dimensions

    # Predict the emotion
    predictions = model.predict(img_resized)
    predicted_label = labels[np.argmax(predictions)]

    # Display the prediction
    print(f"Predicted Emotion: {predicted_label}")
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Emotion: {predicted_label}")
    plt.axis('off')
    plt.show()


X_train, Y_train = preprocess(mode='train')
X_test, Y_test = preprocess(mode='test')
model, history = train_model(X_train, Y_train, X_test, Y_test)
evaluate(X_test, Y_test)
plot_history(history)
plot_confusion_matrix(model, X_test, Y_test, emotion_labels)
# Predict emotion for a user-selected image
predict_emotion(model, emotion_labels)