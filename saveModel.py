import os
import numpy
import pickle
import cv2

os.chdir(r'C:\Users\Vilas Rabad\Desktop\Python\Smart-Intruder-Detection-System-main')
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'Face_Data'

print('Line 9')

(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

width, height = 130, 100

print('Line 25')
(images, labels) = [numpy.array(lst) for lst in [images, labels]]

print('Line 29')
model = cv2.face.LBPHFaceRecognizer_create()
print('Line 32')
model.train(images, labels)

# Save model using pickle
model.save("trained_face_recognizer.xml")

print("Model saved successfully!")
