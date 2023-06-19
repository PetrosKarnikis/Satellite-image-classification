# Import required libraries
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.io import imread_collection, imread
from skimage.transform import resize

# Define the data directory and the image size
data_dir = '../input/satellite-image-classification/data'
img_size = (64, 64)

# Load the image data
image_paths = []
labels = []
for label in os.listdir(data_dir):
    for img_path in os.listdir(data_dir+'/'+label):
        image_paths.append(os.path.join(data_dir, label, img_path))
        labels.append(label)

# Read the images and resize them to the required size
images = imread_collection(image_paths)
resized_images = [resize(img, img_size, anti_aliasing=True) for img in images]

# Convert the images to numpy arrays
X = np.array(resized_images)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the images to create feature vectors
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Define the decision tree classifier and fit the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_flat, y_train)

# Make predictions on the test data and calculate accuracy
y_pred = clf.predict(X_test_flat)
acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {acc:.2f}')
