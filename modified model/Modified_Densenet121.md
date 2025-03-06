---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd 
import pickle
import cv2, PIL, glob, pathlib
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras import backend as K
from tensorflow.keras.optimizers import SGD,RMSprop,Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from subprocess import check_output
import tensorflow as tf
import seaborn as sns
import warnings
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
warnings.filterwarnings('ignore')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
.
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, Dropout, BatchNormalization

```

```python

print(check_output(["ls", "../input/eyediseasedata/dataset"]).decode("utf8"))
print(os.listdir("../input/eyediseasedata/dataset"))

```

```python
# generating dataset from directory

# Generating train dataset
data = tf.keras.utils.image_dataset_from_directory(directory = '../input/eyediseasedata/dataset',
                                                   color_mode = 'rgb',
                                                   batch_size = 64,
                                                   image_size = (224,224),
                                                   shuffle=True,
                                                   seed = 2022)

```

```python
labels = np.concatenate([y for x,y in data], axis=0)
```

```python

```

```python
values = pd.value_counts(labels)
values = values.sort_index()
```

```python
# getting class names
class_names = data.class_names
for idx, name in enumerate(class_names):
  print(f"{idx} = {name}", end=", ")
```

```python
data_iterator = data.as_numpy_iterator()
```

```python
batch = data_iterator.next()
```

```python
batch[0].shape
```

```python
plt.figure(figsize=(10, 10))
for images, labels in data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```

```python
data = data.map(lambda x, y: (x/255, y))
```

```python
sample = data.as_numpy_iterator().next()
```

```python
print(sample[0].min())
print(sample[0].max())
```

```python
print("Total number of batchs = ",len(data))
```

```python
train_size = int(0.7 * len(data)) +1
val_size = int(0.2 * len(data))
test_size = int(0.1 * len(data))
```

```python
train = data.take(train_size)
remaining = data.skip(train_size)
val = remaining.take(val_size)
test = remaining.skip(val_size)
```

```python
print(f"# train batchs = {len(train)}, # validate batchs = {len(val)}, # test batch = {len(test)}")
len(train) + len(val) + len(test)
```

```python

# # add data augmentation to training set
# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.1),
#     ]
# )
```

```python
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(factor=0.1),
        layers.RandomBrightness(factor=0.1),
        layers.GaussianNoise(stddev=0.1),
    ]
)

```

```python
train = train.map(lambda x, y: (data_augmentation(x), y))
```

```python
test_iter = test.as_numpy_iterator()
```

```python
test_set = {"images":np.empty((0,224,224,3)), "labels":np.empty(0)}
while True:
  try:
    batch = test_iter.next()
    test_set['images'] = np.concatenate((test_set['images'], batch[0]))
    test_set['labels'] = np.concatenate((test_set['labels'], batch[1]))
  except:
    break
```

```python
y_true = test_set['labels']

```

```python
dense = DenseNet121(weights = "imagenet", include_top = False, input_shape=(224,224,3))
```

```python
for layer in dense.layers[:121]:
    layer.trainable = False
```

```python
# # old
# model = Sequential()
# model.add(dense)
# model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(1, 1)))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(1, 1)))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(1, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(1, 1)))
# model.add(BatchNormalization())
# model.add(GlobalAveragePooling2D())
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))
```

```python
# from keras.utils.vis_utils import plot_model

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
```

```python
# # old
# model = Sequential()
# model.add(dense)
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(512,activation= "relu"))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation = "sigmoid"))
```

```python
# # 1
# x = dense.output

# x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)

# x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
# x = Dense(512,activation= "relu")(x)
# x = Dropout(0.5)(x)
# pred = Dense(4, activation='softmax')(x)

# model = Model(inputs=dense.input, outputs=pred)

# model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
# 2
x = dense.output

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(1, 1))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(1, 1))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(1, 1))(x)

x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(512,activation= "relu")(x)
x = Dropout(0.5)(x)
pred = Dense(4, activation='softmax')(x)

model = Model(inputs=dense.input, outputs=pred)

model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
# # 3
# x = dense.output

# x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2, strides=(1, 1))(x)
# x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)

# x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
# x = Dense(512,activation= "relu")(x)
# x = Dropout(0.5)(x)
# pred = Dense(4, activation='softmax')(x)

# model = Model(inputs=dense.input, outputs=pred)

# model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
# # 4
# x = dense.output

# x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)

# x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
# x = Dense(512,activation= "relu")(x)
# x = Dropout(0.5)(x)
# pred = Dense(4, activation='softmax')(x)

# model = Model(inputs=dense.input, outputs=pred)

# model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
# # 5
# x = dense.output

# x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), strides=(1, 1))(x)

# x = BatchNormalization()(x)
# x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
# x = Dense(512,activation= "relu")(x)
# x = Dropout(0.5)(x)
# pred = Dense(4, activation='softmax')(x)

# model = Model(inputs=dense.input, outputs=pred)

# model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
from keras import callbacks
file_path = "densenet_best_.hdf5"
checkpoint = ModelCheckpoint(file_path,monitor="val_accuracy", verbose=1, restore_best_weights=True, mode="max")
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)
callbacks_list = [checkpoint, early]
```

```python
history = model.fit_generator     
    train,
    validation_data=val,
    epochs=50,
    callbacks = callbacks_list 
    )
```

```python
model.evaluate(test)
```

```python
# model.summary()
```

```python
# from keras.utils.vis_utils import plot_model

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
```

```python
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
# !pip install visualkeras
# import visualkeras
```

```python
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_his(history):
    plt.figure(figsize=(15,12))
    metrics = ['accuracy', 'loss']
    for i, metric in enumerate(metrics):
        plt.subplot(220+1+i)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    plt.show()
```

```python
plot_his(history)
```

```python
y_pred = np.argmax(model.predict(test_set['images']), 1)
```

```python
print(classification_report(y_true, y_pred, target_names = class_names))
```

```python
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
plt.xticks(np.arange(4)+.5, class_names, rotation=90)
plt.yticks(np.arange(4)+.5, class_names, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
```

```python
# model.summary()
```

```python
# model.save_weights('densmodel_weights.h5')
# model.save('densmodel_keras.h5')
```

```python
# from keras.utils.vis_utils import plot_model
# plot_model(model)
```
