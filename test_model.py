import numpy as np
import pandas as pd
import os
from keras.models import load_model
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
import pyttsx3

model_path = './traffic_sign_detection_model.keras'
print(f"Loading model from: {model_path}")
model = load_model(model_path)

# def testing(testcsv):
#     cur_path = os.getcwd()
#     base_dir = os.path.join(cur_path, 'dataset')
#     print(f"Base directory for images: {base_dir}")

#     print(f"Reading test CSV file: {testcsv}")
#     y_test = pd.read_csv(testcsv)
#     label = y_test["ClassId"].values
#     imgs = y_test["Path"].values
#     print(f"Number of test samples found: {len(imgs)}")

#     data = []
#     for idx, img in enumerate(imgs):
#         img_path = os.path.join(base_dir, img)
#         try:
#             print(f"Processing {idx + 1}/{len(imgs)}: {img_path}")
#             # Open and resize the image
#             image = Image.open(img_path)
#             image = image.resize((30, 30))
#             data.append(np.array(image))
#         except FileNotFoundError:
#             print(f"[Warning] File not found: {img_path}")
#             continue
#         except Exception as e:
#             print(f"[Error] Unable to process image {img_path}: {e}")
#             continue

#     X_test = np.array(data)
#     print(f"Finished processing images.")
#     return X_test, label

# X_test, label = testing('./dataset/Test.csv')

# if X_test.size > 0:
#     print("Making predictions on the test dataset")
#     Y_pred = np.argmax(model.predict(X_test, verbose=1), axis=-1)

#     print("Predictions completed. Calculating accuracy...")
#     print("Predicted labels:", Y_pred)
#     print("True labels:", label[:len(Y_pred)])
#     print(f"Accuracy: {accuracy_score(label[:len(Y_pred)], Y_pred):.4f}")
# else:
#     print("No valid data available for prediction.")

# the 43 different classes for classification
CLASSES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
    'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection', 
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 
    'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 
    'Keep left', 'Roundabout mandatory', 'End of no passing', 
    'End no passing veh > 3.5 tons'
]

# function for text to speech
def Speak (txt):
    engine = pyttsx3.init()
    engine.setProperty('rate', 90)
    engine.setProperty('volume', 1)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    text = txt
    engine.say(text)
    engine.runAndWait()

# function for testing an image    
def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict(X_test)
    predicted_class = np.argmax(Y_pred)
    return image,predicted_class

plot,prediction = test_on_img(r'C:\Users\LENOVO\Desktop\traffic sign detection\dataset\Test\00500.png')
print("Predicted traffic sign is: ", CLASSES[prediction])
Speak(CLASSES[prediction])
plt.imshow(plot)
plt.show()