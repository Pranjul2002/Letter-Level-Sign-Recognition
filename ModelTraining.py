
# For Training model

import os
import random
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

data_dir= "Train_Data"
IMG_SIZE=64
BATCH_SIZE=1
EPOCH=25

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

data={}
img_path=[]
Sign=[]

def create_df(data_dir):
    for label in os.listdir(data_dir):
        label_path=os.path.join(data_dir, label)
        for instanceOfSign in os.listdir(label_path):
            img_path.append(os.path.join(label_path, instanceOfSign))
            Sign.append(label)


create_df(data_dir)

df=pd.DataFrame({'img_path':img_path, 'sign':Sign})
#print(df)

labels=sorted(os.listdir(data_dir))
labels_index={labels[i]: i for i in range(len(labels))}
print(labels_index)

index_labels=dict([(labels_index[key], key) for key in labels_index])
print(index_labels)

df["index"]=df["sign"].map(labels_index)
#print(df)

X=df["img_path"]
y=df["index"]


def preprocess_images(X):
    X_processed = []
    for path in X:
        img = cv.imread(path)
        if img is not None:
            img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # normalize
            X_processed.append(img)
    return np.array(X_processed)

# Preprocess images and convert labels to NumPy array
X_processed = preprocess_images(X)
y_array = np.array(y)
y_encoded = to_categorical(y_array, num_classes=len(labels_index))  # one-hot encoding

print(y_array)
print(y_encoded)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y_encoded, test_size=0.4, random_state=42)

model=models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Dropout(0.5),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(labels_index), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train model
model_history=model.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(model_history.history['loss'], label='Train Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# Save model (optional)
model.save("sign_classifier_model4.h5")


