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

```python _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
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
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,Average
```

```python
model_1=load_model('../input/models/model/efficient_best_.hdf5')
```

```python
model_1=Model(inputs=model_1.inputs,
             outputs=model_1.outputs,
             name='Effiecent')
```

```python
model_2=load_model('../input/models/model/inception_best.hdf5')
```

```python
model_2=load_model('../input/models/model/inception_best.hdf5')
```

```python
model_2=Model(inputs=model_2.inputs,
             outputs=model_2.outputs,
             name='Inception')
```

```python
model_3=load_model('../input/models/model/densenet_best_.hdf5')
```

```python
model_3=Model(inputs=model_3.inputs,
             outputs=model_3.outputs,
             name='Dense')
```

```python
models=[model_1,model_2,model_3]
model_input=Input(shape=(224,224,3))
model_outputs=[model(model_input) for model in models]
```

```python
class WeightedAverageLayer(tf.keras.layers.Layer):
    def __init__(self,w1,w2,w3,**kwargs):
        super(WeightedAverageLayer,self).__init__(**kwargs)
        self.w1=w1
        self.w2=w2
        self.w3=w3
    
    def call(self, inputs):
        return self.w1*inputs[0] + self.w2*inputs[1] + self.w3*inputs[2]
```

```python
ensemble_output= WeightedAverageLayer(0.3, 0.4 ,0.3)(model_outputs)
ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
```

```python
ensemble_model.compile(optimizer = 'adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
ehistory = ensemble_model.fit(
    train,
    validation_data=val,
    epochs=50
    )
```

```python
ensemble_model.evaluate(test)
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
plot_his(ehistory)
```

```python
y_pred = np.argmax(ensemble_model.predict(test_set['images']), 1)
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
ensemble_model.summary()
```

```python
#Model Save
# ensemble_model.save_weights('enmodel_weights.h5')
# ensemble_model.save('enmodel_keras.h5')
```
