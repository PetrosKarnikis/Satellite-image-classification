import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator

# Set up paths and parameters
data_dir = '../input/satellite-image-classification/data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
num_classes = len(os.listdir(train_dir))
img_size = 64

# Create a data generator for image augmentation
datagen = ImageDataGenerator(rescale=1./255)

# Load training data and labels
train_data = datagen.flow_from_directory(train_dir, 
                                         target_size=(img_size, img_size), 
                                         batch_size=32, 
                                         class_mode='categorical', 
                                         shuffle=True)
train_images, train_labels = next(train_data)
for images, labels in train_data:
    train_images = np.concatenate([train_images, images])
    train_labels = np.concatenate([train_labels, labels])
    if len(train_labels) >= len(train_data.classes):
        break

# Load test data and labels
test_data = datagen.flow_from_directory(test_dir, 
                                        target_size=(img_size, img_size), 
                                        batch_size=32, 
                                        class_mode='categorical', 
                                        shuffle=False)
test_images, test_labels = next(test_data)
for images, labels in test_data:
    test_images = np.concatenate([test_images, images])
    test_labels = np.concatenate([test_labels, labels])
    if len(test_labels) >= len(test_data.classes):
        break

# Flatten the images for SVM input
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Train an SVM classifier
clf = SVC(kernel='linear', C=1.0, probability=True)
clf.fit(train_images_flat, np.argmax(train_labels, axis=1))

# Evaluate the SVM classifier
train_preds = clf.predict(train_images_flat)
train_acc = accuracy_score(np.argmax(train_labels, axis=1), train_preds)

test_preds = clf.predict(test_images_flat)
test_acc = accuracy_score(np.argmax(test_labels, axis=1), test_preds)

print('Training accuracy:', train_acc)
print('Test accuracy:', test_acc)
