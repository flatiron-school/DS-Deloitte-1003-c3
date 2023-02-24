# Convolution Neural Networks and COVID-19 Omicron and Delta Variant Lung CT Scans

## Data & Kaggle

### About the dataset

While the COVID-19 pandemic was waning in most parts of the world, a new wave of COVID-19 Omicron and Delta variants in central Asia and the Middle East caused a devastating crisis and collapse of health care systems. As the diagnostic methods for this COVID-19 variant became more complex, healthcare centers faced a dramatic increase in patients. Thus, the need for less expensive and faster diagnostic methods led researchers and specialists to work on improving diagnostic testing.

This is a large public COVID-19 (Omicron and Delta Variant) lung CT scan [dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/covid19-omicron-and-delta-variant-ct-scan-dataset). it contains 14,482 CT scans which include 12,231 positive cases (COVID-19 infection) and 2251 negative ones (normal and non-COVID-19). Data is available as 512×512px JPG images and have been collected from patients in radiology centers of teaching hospitals of Tehran, Iran. 

### Google Colab

Google [Colab](https://colab.research.google.com/) is Google notebook with features of visual studio. We will be using Google Colab for this lab for a few reasons.

*   If you are not familar with , you should become so.
*   It is easier to bring in the data from [Kaggle](https://www.kaggle.com/) (see below) via Colab, particularly when students have a myriad of types of computers and plethora of set-ups on those computers.
*   It can connect directly to GitHub.



### Kaggle

While the data set could be downloaded from Kaggle via your browser and then you could upload all of the data into your notebook, that is not efficient since the data is over 1.5 GB. Thus, we want to connect to Kaggle so that we can download the data diectly. Here is the process.

1.   Go to kaggle.com and log-in or create an account.
2.   On the upper tab, click on 'Account'.
3.   Once you do that, you'll see *API* and below that, "Create New API Token."
4.   After clicking the "Create New API Token," a file named "kaggle.json" will be downloaded.
5.   Upload this file into your Colab notebook, just as you would with a data set.

N.B. You can reuse the same .json file, you don't need to create a new API token each time you want to connect to Kaggle.

*Before proceeding to the code immediately below, make sure to place kaggle.json in your Colab notebook Files.*


```python
# We are using Unix commands (each begins with '!') to connect to Kaggle and get the data.

# Set-up the Kaggle directory
!mkdir -p ~/.kaggle

# Copy the json file to this new directory.
!cp kaggle.json ~/.kaggle/

# Allow access to the directory
!chmod 600 ~/.kaggle/kaggle.json

# List the names of the files in the directory
!ls ~/.kaggle
```

    kaggle.json



```python
# Install Kaggle packages

!pip install -q kaggle
!pip install -q kaggle-cli

# Download the data set
!kaggle datasets download -d mohammadamireshraghi/covid19-omicron-and-delta-variant-ct-scan-dataset

# Remove the working directory
!rm -rf /kaggle/working/*

# If you get any errors, its likely due to conflicts in the Python versions and the Unix versions,
# but they should not be an issue. They are more warnings, then errors.
# As long as it downloads the data, you're fine.
```

    [K     |████████████████████████████████| 74 kB 1.8 MB/s 
    [K     |████████████████████████████████| 4.2 MB 31.4 MB/s 
    [K     |████████████████████████████████| 112 kB 55.3 MB/s 
    [K     |████████████████████████████████| 147 kB 69.9 MB/s 
    [K     |████████████████████████████████| 50 kB 5.6 MB/s 
    [?25h  Building wheel for kaggle-cli (setup.py) ... [?25l[?25hdone
      Building wheel for lxml (setup.py) ... [?25lerror
    [31m  ERROR: Failed building wheel for lxml[0m
    [?25h  Building wheel for PrettyTable (setup.py) ... [?25l[?25hdone
      Building wheel for pyperclip (setup.py) ... [?25l[?25hdone
        Running setup.py install for lxml ... [?25l[?25herror
    [31mERROR: Command errored out with exit status 1: /usr/bin/python3 -u -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-hnlxnhyu/lxml_c7afa52f268949efad0c45038e335816/setup.py'"'"'; __file__='"'"'/tmp/pip-install-hnlxnhyu/lxml_c7afa52f268949efad0c45038e335816/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record /tmp/pip-record-q7i_yxkp/install-record.txt --single-version-externally-managed --compile --install-headers /usr/local/include/python3.7/lxml Check the logs for full command output.[0m
    Downloading covid19-omicron-and-delta-variant-ct-scan-dataset.zip to /content
    100% 1.55G/1.55G [00:09<00:00, 187MB/s]
    100% 1.55G/1.55G [00:09<00:00, 167MB/s]



```python
# Unzip the data

!unzip covid19-omicron-and-delta-variant-ct-scan-dataset.zip
```

### Import the germane libraries


```python
# Import libraries

import numpy as np
import pandas as pd

import pathlib
import PIL

# import your machine learning libraries here 

import os
import datetime
import time
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
```

## Preprocessing and EDA

### Set-up the folders and count the data


```python
# Create folders for the two types of data.

data_dir_covid = pathlib.Path('../content/COVID19_Omicron_and_Delta_CT_Scans_dataset/COVID')
data_dir_non_covid = pathlib.Path('../content/COVID19_Omicron_and_Delta_CT_Scans_dataset/Non_COVID')
```


```python
# Count the number of .jpg files in each folder.
img_count_covid = len(list(data_dir_covid.glob('*.jpg'))) 
img_count_non_covid = len(list(data_dir_non_covid.glob('*.jpg'))) 
```


```python
# Print the image counts
print("Image count in Covid set: ",img_count_covid)
print("Image count in Non Covid set: ",img_count_non_covid)
print("Total Image count: ",(img_count_covid+img_count_non_covid))
```

### Viewing some images


```python
# Let's go back to the two different types of data.
# And let's look at a couple of each type.

covid = list(data_dir_covid.glob('*'))
non_covid = list(data_dir_non_covid.glob('*'))
```


```python
# Images
# Images
img1 = PIL.Image.open(str(covid[0]))
print("Image size: " ,img1.size)
img1
```


```python
img2 = PIL.Image.open(str(non_covid[0]))
print("Image size: " ,img1.size)
img2
```

Feel free to look at a few more images, and try to determine if you can tell any difference between non-COVID and COVID lung scans.

### Train/Test/Validation

We'll now split the data into train/test/validaiton sets using `splitfolders`.


```python
# We'll use split-folders to combine the two types of data (Covid/Non-Covid)
# and to further split into train/test/validaiton sets.

# Install split-folders
!pip install split-folders

# Using Split-folders to split source folder into the
# train (70%), test (20%), and validation (10%).

# Set the seed to 1882, so that we can replicate the results.

import splitfolders
splitfolders.ratio("../content/COVID19_Omicron_and_Delta_CT_Scans_dataset", output="../working/dataset",
    seed=1882, ratio=(.7, .2, .1), group_prefix=None, move=False)
```

Now let's define a path for of the three sets, and like before count the images and print to verify all has gone correctly.


```python
# Define the path for train, validation and test set

data_dir_train = pathlib.Path('../working/dataset/train')
data_dir_test = pathlib.Path('../working/dataset/test')
data_dir_val = pathlib.Path('../working/dataset/val')

# Check the total image counts (all images are of type .png).

img_count_train = len(list(data_dir_train.glob('*/*.jpg'))) 
img_count_test = len(list(data_dir_test.glob('*/*.jpg'))) 
img_count_val = len(list(data_dir_val.glob('*/*.jpg'))) 

img_count_tot = img_count_train + img_count_test + img_count_val

print("Image count in Train set: ",img_count_train)
print("Image count in Val set: ",img_count_val)
print("Image count in Test set: ",img_count_test)
print("Total image count",img_count_tot)
```

### Reshaping the data

Since the data images are 512 by 512, we need to rescale the data.


```python
# Reshape the data

train_gen = ImageDataGenerator(rescale=1./511).flow_from_directory(
    data_dir_train,
    target_size = (128,128),
    batch_size = 10136)

val_gen = ImageDataGenerator(rescale=1./511).flow_from_directory(
    data_dir_val,
    target_size = (128,128),
    batch_size = 2896)

test_gen = ImageDataGenerator(rescale=1./511).flow_from_directory(
    data_dir_test,
    target_size = (128,128),
    batch_size = 1450)
```

Finally, we need to have images and their labels.


```python
# Create the data sets
train_images, train_labels = next(train_gen)
test_images, test_labels = next(test_gen)
val_images, val_labels = next(val_gen)
```

## Modeling

### Data Generator

Before we get to convolution neural networks, we need to set-up the machinery for the CNN.

A data generator allows Python to be more effiecent in reading the data. This is particular important for visual data. While Keras has built in data generator, from `tensorflow.keras.preprocessing.image.ImageDataGenerator`, it has limited flexibility. So, we'll create our own.


```python
 # Custom data generator

def data_generator(data_source,img_height, img_width, btc_size):    
    return tf.keras.utils.image_dataset_from_directory(
        data_source,
        validation_split=None, # We already split the data
        subset=None,
        seed=123,
        color_mode='grayscale',
        image_size=(img_height, img_width),
        batch_size=btc_size,
        crop_to_aspect_ratio=True,
        shuffle=True
    )
```

Set the inital values for the batch size and the number of epochs.

Be careful/patient with the bactch size and the number of epochs... this may take quite some time to run depending on your choices.


```python
# We now need to set the (initial) values of the hyperparameters.
batch_size = 'none'
img_height = 256
img_width = 256
num_epochs = 'none'
```

Use `data_generator` on the three subsets of the data.


```python
# Applying data_generator

train_ds = data_generator(data_dir_train,img_height, img_width, batch_size)
val_ds = data_generator(data_dir_val,img_height, img_width, batch_size)
test_ds = data_generator(data_dir_test,img_height, img_width, batch_size)
```


```python
# We'll need the class names

class_names = train_ds.class_names
print(class_names)
```

### Some further model preperation

A few more items to take care of before we train our model.

We'll start with our number of classes.


```python
# Number of classes
num_classes = len(class_names)
```

We need to make sure our layers are scaled as well.


```python
# Rescaling 
normalization_layer = layers.Rescaling(1./255)
```

To make life easier, let's define a function that will print the parameters for us.


```python
# Function for printing parameters
def print_param():
    print("*** Params used in Model Training ****")
    print("Batch Size: ", batch_size)
    print("Epoch Size: ", num_epochs)
    print("Image size: {} {}".format(img_height, img_width))
    print("***********************")
```

Defining our training model


```python
# model_init: str, which is a string to be prefixed in model checkpoint name.

def train_model(model_init, model):
    
    #Create training and validation sets
    train_ds = data_generator(data_dir_train,img_height, img_width, batch_size)
    val_ds = data_generator(data_dir_val,img_height, img_width, batch_size)
    test_ds = data_generator(data_dir_test,img_height, img_width, batch_size)

    # File name for model checkpoint
    curr_dt_time = datetime.datetime.now()
    model_name = model_init + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{accuracy:.5f}-{val_loss:.5f}-{val_accuracy:.5f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    # Stop Training, if no improvement observed.
    Earlystop = EarlyStopping( monitor="val_loss", min_delta=0,patience=7,verbose=1)
    
    # Reduce learning rate when performance metric stopped improving.
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5,
                           cooldown=4, verbose=1,mode='auto',epsilon=0.0001)
    
    callbacks_list = [checkpoint, LR, Earlystop]        
    
    # Print parameters used for model training using the functiond defined above.
    print_param()
    
    start = time.time()
    history = model.fit(train_ds, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_ds, 
                    class_weight=None, workers=1, initial_epoch=0)
    end = time.time()
    print("Total training time: ", "{:.2f}".format((end-start)), " secs")
    return history
```

Creating a another definition so that we can print the metrics with ease.


```python
# Plot metrics function

def plot_metrics(history):       
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(num_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
```

### Define and compile the model

We are finally ready to go. Time for you to define your model and then compile it.

Since this is a large image data set, the model will possibly take some time to run. Make sure to use regularization techniques to guard against overfitting (variance) and to reduce the runtime.

As a reminder, in `train_model` above, we have already included both `EarlyStopping` and `ReduceLROnPlateau`.


```python
# Define the model

model1 = Sequential([
 
 # Build your CNN here

    ])
```

Give the model summary


```python
# Summary of the model

model1.summary()
```

Compile the model


```python
# Compile the model

model1.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

```

Set the inital hyperparamters to experiment.



```python
# Setting hyperparameters to experiment

batch_size= 'none'
img_height = 256
img_width = 256
num_epochs = 'none'
```

Let's train the model.


```python
history_model1 = train_model("model1",model1)
```

Let's plot the metrics.


```python
# Plot metrics
num_epochs = len(history_model1.history['loss'])
plot_metrics(history_model1)
```

### Model accuracy on the test set

So how did you do?


```python
pred = model1.predict(test_ds)
bin_predict = np.argmax(pred,axis=1)
```


```python
test_loss, test_acc = model1.evaluate(test_ds, batch_size=batch_size, verbose=2)
```

Continue to modify your model until you are happy with it, i.e. when you find the optimal model.
