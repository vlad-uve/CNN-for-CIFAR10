# Implementation of CNN to Classify CIFAR-10 Images
## Project Overview

This project was develoed as part of **the YCBS 258 - Practical Machine Learning at Mcgill University**. The objective was to implement a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using the Google Colab platform. The project aimed to achieve a target model accuracy of at least 0.8. 

The model architecture was inspired by AlexNet to achieve high performance, chosen for its proven effectivness in image classification tasks, however, with adjustments and simplifications made to fit the capacity constrains of the platform and the scope of the problem. To further enhance model performance and mitigate overfitting, several regularization techniques were applied. This implementation demonstrates the application of deep learning techniques for image classification tasks, emphasizing both architectural adaptation and performance optimization within resource limitations.

## Dataset Description

The model was trained and evaluated on the CIFAR-10 dataset, a widely used benchmark collection of color images for classification tasks. It consists of 60,000 color images, each 32x32 pixels in size, encoded with 3 color channels (RGB). The dataset is divided into 50,000 training images and 10,000 testing images. The images are equally distributed among 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each class contains 5,000 training samples and 1,000 testing samples. For more details, refer to the CIFAR-10 official page (https://www.cs.toronto.edu/~kriz/cifar.html).

### Loading the CIFAR-10 Dataset
For this project, the CIFAR-10 dataset was downloaded directly from Keras (https://keras.io/api/datasets/cifar10/). 
```python
from tensorflow import keras

#Load train and test CIFAR-10 data set from Keras
#x_train and y_train are the training data set
#x_test and y_test are the testing data set
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```
### Exploring the Shapes
The dataset consists of both images and labels, which are stored in arrays with the following shapes:
* Training data (x_train): A NumPy array of shape (50000, 32, 32, 3), where 50,000 images are 32x32 pixels in size with 3 color channels (RGB).
* Training labels (y_train): A NumPy array of shape (50000, 1), containing the labels for each image (0 to 9 for the 10 classes).
* Test data (x_test): A NumPy array of shape (10000, 32, 32, 3), containing the 10,000 test images, also 32x32 pixels with 3 color channels.
* Test labels (y_test): A NumPy array of shape (10000, 1), containing the corresponding labels for the test set.

WRITE ABOUT SHAPES and INPUT AND OUTPUT SIZES!!!

```python
#Print shape of the training data set
print('Shape of the training image set: {}'.format(x_train.shape)) #(50000, 32, 32, 3)
print('Shape of the training classe set: {}'.format(y_train.shape)) #(50000, 1)

#Print shape of the testing data set
print('Shape of the testing image set: {}'.format(x_test.shape)) #(10000, 32, 32, 3)
print('Shape of the testing classe set: {}'.format(y_test.shape)) #(10000, 1)
```

### Displaying Sample Images
To better understand the dataset, the following visualization displays the first 5 sample images of each class from the CIFAR-10 training set, along with their corresponding class labels.
![image](https://github.com/user-attachments/assets/a9cd3358-5b42-4aef-9655-25165692d17c)

### Preprocessing
In this step, to ensure optimal training performance, CIFAR-10 dataset is prepared for training by performing the following key operations:

1. Normalizing the Image Data: The pixel values of the images are scaled from the range [0, 255] to [0, 1]. This step improves model convergence and stability during training.
```python
#Data normalization
x_train=x_train/255. #normalaized training image set
x_test=x_test/255. #normalaized testing image set
```
2. One-Hot Encoding of Class Labels: The CIFAR-10 labels are integers that represent the 10 classes (e.g., 0 for airplane, 1 for automobile, etc.). However, neural networks require the class labels to be in a binary format, hence, the labels are concerted into a binary matrix using one-hot encoding. This prevents the model from assuming an ordinal relationship between the classes and ensures that the model learns to distinguish each class independently.
```python
#Convert class integers into binary matrix
y_train=to_categorical(y_train,10) #encoded training class set
y_test=to_categorical(y_test,10) #encoded testing class set
```












## Model Created

A CNN model was defined using the Keras Functional API, with an architecture inspired by AlexNet to enhance performance (https://en.wikipedia.org/wiki/AlexNet#:~:text=AlexNet%20is%20the%20name%20of,D.). Since the CIFAR-10 images are lower in resolution than those for classification of which AlexNet model was initially developped, and in the view that the platform, Google Collab, provides ristricted capacities, the CNN atchitecture was subjected to certain changes aimed to reduce the model complexity and size: 

* The kernel size is reduced to 5x5 and 3x3 windows (resolution is lower and data set size)
* The number of filters is reduced to 32, 96 and 256
* The stride size is 1
* The number of convolution layers is reduced to 4
* The number of fully conncected (dense) layers is reduced to 2

The following regularization methods were used to avoid overfitting and to enhance accuracy of the model:
* Batchnormalization
* Drop out

Hyperparameters:
* Gradient??


```python
def CNN_model():

  global tensorboard_callback, earlyStopping_callback

  #Clean logs
  !rm -rf ./logs/

  #Clear previous session
  clear_session()

  #Define the input layer suitable for 32x32 RGB images
  inputs=Input(shape=(32,32,3), name='Input_Data')

  #Define convolution layers below
  #First convolution layer with 32 filters, 5x5 kernel size, and 1x1 strides
  t=Conv2D(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu', name='Conv_2D_1' )(inputs)
  #First batch normalization layer
  t=BatchNormalization(name='Batch_Norm_1')(t)

  #Second convolution layer with 96 filters, 3x3 kernel size, and 1x1 strides
  t=Conv2D(filters=96, kernel_size=3, strides=1, padding="same", activation='relu', name='Conv_2D_2')(t)
  #Second batch normalization layer
  t=BatchNormalization(name='Batch_Norm_2')(t)

  #First max pooling layer with 3x3 pool size and 2x2 strides
  t=MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='Max_Pool_1')(t)
  #First drop out layer
  t=Dropout(rate=0.4, name='Drop_Out_1')(t)

  #Third layer of convolution with 256 filters, 3x3 kernel size, and 1x1 strides
  t=Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv_2D_3')(t)
  #Third batch normalization layer
  t=BatchNormalization(name='Batch_Norm_3')(t)

  #Forth layer of convolution with 96 filters, 3x3 kernel size, and 1x1 strides
  t=Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv_2D_4')(t)
  #Forth batch normalization layer
  t=BatchNormalization(name='Batch_Norm_4')(t)


  #Second max pooling layer with 3x3 pool size and 2x2 strides
  t=MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='Max_Pool_2')(t)
  #Second drop out layer
  t=Dropout(rate=0.4, name='Drop_Out_2')(t)

  #Define first fully connected layer
  #Flatten layer
  y1=Flatten(name='Flatten_y1')(t)
  #Dense layer
  y1=Dense(4096, activation='relu', kernel_initializer='glorot_uniform', name='Dense_y1')(y1)
  #Drop out layer
  y1=Dropout(rate=0.4, name='Drop_Out_y1')(y1)

  #Define second fully connected layer
  #Flatten layer
  y2=Flatten(name='Flatten_y2')(y1)
  #Dense layer
  y2=Dense(4096, activation='relu', kernel_initializer='glorot_uniform', name='Dense_y2')(y2)
  #Drop out layer
  y2=Dropout(rate=0.4, name='Drop_Out_y2')(y2)

  #Define output layer and model
  outputs=Dense(10, activation='softmax', name='Output_Model')(y2)
  model=Model(inputs, outputs)

  #Initialize Tensor Board - logs
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  #Define callbacks
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1)
  earlyStopping_callback = EarlyStopping(monitor='val_accuracy', patience=10)


  #Compile Model
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])

  return model
 ```    

![image](https://github.com/user-attachments/assets/c8a52f97-b356-4b55-940b-bec793cd5d27)



## Model Trained


![image](https://github.com/user-attachments/assets/d03dfe7b-e4ed-4786-9509-2ece25337b73)





## Model Tested

Testing Accuracy = 0.83
Testing Loss = 0.80


![image](https://github.com/user-attachments/assets/42de68fe-8dd8-46ab-b1ae-115cb844a83e)



## Results


Tips:
- Start with only a few of a layers and check if the network can learn.
- Add units and layers progressively.
- Kernels that are too large or too much pooling will reduce the size of layer ouputs
- Try Batch Norm and Dropout
- If you don't reproduce the exact architecture, that is fine. Explain what you changed and why!.
- Functional API!

As the size of the given data is much less than the AlexNet's input data, i.e., the given images have size 32x32x3 whileas the AlexNet's input is 227x227x3, hence, significant simplification is required. First of all, I reduced the first convolution layer's kernel size from 11x11 to 5x5. Because this layer serves to heavily reduce the input data's dimension, but in our case, it is not required. Similarly, I used 3x3 kernel in the second convolution layer due to the same reason. The smaller kernel size should be a good simplification.

Morover, the complexity of the given data is much less than that used as an input in the classical AlexNet archtecture. Therefore, the number of frames was reduced as well, 32 vs 96, and 96 vs 256, and 129 vs or 384. size in the second convlolution -the data input was already in smaller in size.

As the given input is much closer to the output of the classical AlexNet, the padding and convolution were applied without high strides, and padding was used just once to fit the sizes of the AlexNet outputs.

Finally, I used kernel 1x1 and only 1 convolution in the last convolution sequance to seriously minimize number of model parameters and computational time. Fully conected layers sizes and their quontity had also be simlified to respect lesser complexity of the model and reduce necessary computational perfromance.

During the training stage, I realized that the usage of batch normalization and dropout rate was necessary to improve learning speed and handlle with overfitting that apparently occurs because of the simplicity. The achieved acuracy is 0.8091, validation accuracy is 0.7988 and test accuracy 0.7797. As the difference between the accuracy values is not significant, hence, the overfitting was avoided.

"Adam" optimizer was chosen as it demonstrated better performance in terms of val_Accuracy. Moreover, a model with smaller batch size showed better results. Furthermore, schedules for learning rate were used to assit the model disrupt a plateau of constant accuracy.


