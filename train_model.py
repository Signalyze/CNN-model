import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train_model(data_path, labels_path, save_path, epochs=20):
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    print(f"Loading labels from: {labels_path}")
    labels = np.load(labels_path)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)

    # code for creating a sequential CNN model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    print("Compiling the model")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1  
    )

    print(f"Training completed. Saving model to: {save_path}")
    model.save(save_path)
    print("Model saved successfully.")

    return history

DATA_PATH = './training/data.npy'
LABELS_PATH = './training/labels.npy'
MODEL_SAVE_PATH = './traffic_sign_detection_model.keras'
EPOCHS = 20

history = train_model(DATA_PATH, LABELS_PATH, MODEL_SAVE_PATH, epochs=EPOCHS)