import os
import numpy as np
from PIL import Image

def preprocess_data(classes, dataset_path, save_dir):
    data = []
    labels = []

    for i in range(classes):
        path = os.path.join(dataset_path, str(i))
        print(f"\nProcessing class {i} with path: {path}")
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist. Skipping class {i}.")
            continue

        images = os.listdir(path)
        for a in images:
            try:
                img_path = os.path.join(path, a)
                image = Image.open(img_path)
                image = image.resize((30, 30)) # resizing image to 30 x 30
                data.append(np.array(image))
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {a}: {e}")

    data = np.array(data)
    labels = np.array(labels)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'data'), data)
    np.save(os.path.join(save_dir, 'labels'), labels)
    print("Data and labels saved successfully.")

CLASSES = 43 
DATASET_PATH = './dataset/Train'  
SAVE_DIR = './training'  
preprocess_data(CLASSES, DATASET_PATH, SAVE_DIR)