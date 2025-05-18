# scr/preprocessing.py

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_images_from_folder(base_path, image_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted(os.listdir(base_path))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    
    for cls in class_names:
        class_folder = os.path.join(base_path, cls)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(class_to_idx[cls])

    return np.array(images), to_categorical(labels, num_classes=len(class_names))
