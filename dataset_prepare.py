import cv2
import numpy as np
import os
import re
import shutil


def extract_class(text):
    clipped_text = re.split('(\d+)', text)
    print(clipped_text)
    if clipped_text== ['.DS_Store']:
        return '20'
    else:
        return clipped_text[1]


# data_path = 'data'
data_path = '../datasets/UTKface_inthewild'
category = ["0-10", "10-20", "20-30", "30-40", "40-60", "60-above"]
for label in category:
    if not os.path.exists('dataset/' + label):
        os.makedirs('dataset/' + label)
for folder in os.listdir(data_path):

    for file in os.listdir(data_path + '/' + folder):
        print(folder, file)
        class_id = int(extract_class(file))

        # print(type(class_id))

        source = data_path + '/' + folder + '/' + file
        if class_id < 10:
            destination = 'dataset/' + category[0]
            # print(destination)

            shutil.copy(source, destination)
        elif class_id < 20 and class_id >= 10:
            destination = 'dataset/' + category[1]

            shutil.copy(source, destination)

        elif class_id < 30 and class_id >= 20:
            destination = 'dataset/' + category[2]

            shutil.copy(source, destination)

        elif class_id < 40 and class_id >= 30:
            destination = 'dataset/' + category[3]

            shutil.copy(source, destination)

        elif class_id < 60 and class_id >= 40:
            destination = 'dataset/' + category[4]

            shutil.copy(source, destination)

        elif class_id >= 60:
            destination = 'dataset/' + category[5]

            shutil.copy(source, destination)