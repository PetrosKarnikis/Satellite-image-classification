import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

# Define the data directory
data_dir = '../input/satellite-image-classification/data'

# Get the list of class labels
labels = os.listdir(data_dir)

# Load the image data and corresponding labels
X = []
y = []
for i, label in enumerate(labels):
    label_dir = os.path.join(data_dir, label)
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = imread(image_path)
        image = resize(image, (64, 64))
        X.append(image.flatten())
        y.append(i)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set and evaluate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)
