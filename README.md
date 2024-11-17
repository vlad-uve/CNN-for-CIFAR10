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
The downloaded dataset consists of both images and labels, which are stored in arrays with the following shapes:
* Training data (x_train): A NumPy array of shape (50000, 32, 32, 3), where 50,000 images are 32x32 pixels in size with 3 color channels (RGB).
* Training labels (y_train): A NumPy array of shape (50000, 1), containing the labels for each image (0 to 9 for the 10 classes).
* Test data (x_test): A NumPy array of shape (10000, 32, 32, 3), containing the 10,000 test images, also 32x32 pixels with 3 color channels.
* Test labels (y_test): A NumPy array of shape (10000, 1), containing the corresponding labels for the test set.

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
2. One-Hot Encoding of Class Labels: The CIFAR-10 labels are integers that represent the 10 classes (e.g., 0 for airplane, 1 for automobile, etc.). However, neural networks require the class labels to be in a binary format. To achieve this, the labels are converted into a binary matrix using one-hot encoding. Each label is represented as a binary vector with 10 elements, corresponding to the 10 classes in the CIFAR-10 dataset. In this vector, a 1 is placed in the position of the correct class, and all other positions are filled with 0s. This approach prevents the model from assuming any ordinal relationship between the classes and ensures it learns to distinguish each class independently. As a result, the new shape of the encoded Training Labels set and Test Labels set are (50000, 10) and (10000, 10), respectively.
```python
#Convert class integers into binary matrix
y_train=to_categorical(y_train,10) #encoded training class set
y_test=to_categorical(y_test,10) #encoded testing class set

#Print new shape of training and testing class sets
print('New shape of the training class set: {}'.format(y_train.shape)) #(50000, 10)
print('New shape of the testing class set: {}'.format(y_test.shape)) #(1000, 10)
```
## Model Architecture
For this project, we implemented a Convolutional Neural Network (CNN) inspired by a simplified version of AlexNet (https://en.wikipedia.org/wiki/AlexNet#:~:text=AlexNet%20is%20the%20name%20of,D.), tailored for efficient image classification on the CIFAR-10 dataset. Given that CIFAR-10 images have a lower resolution than those used in AlexNet’s original classification tasks, the CNN architecture was modified to reduce model complexity and size. The design carefully balanced model performance with computational efficiency, taking into account the limited resources of the Google Colab platform. Key modifications include:
* Reduced kernel sizes
* Fewer number of filters
* Small stride size
* Fewer convolutional layers
* Fewer fully connected layers

The following regularization methods were incorporated to minimize overfitting and enhance the model's accuracy:
* Batch Normalization
* Dropout
* Early Stopping

The CNN model was defined using the Keras Functional API, it consists of multiple layers, including convolutional layers, activation functions, pooling layers, and fully connected layers. Below is a detailed overview of the model architecture:

### Input Layer
The input to the model is a batch of images with a shape of (32, 32, 3), i.e., 32x32 pixels with 3 color channels, RGB, as specified by the CIFAR-10 dataset and seen from the training image and testing image sets.

```python
from tensorflow.keras.layers import Input

#Define the input layer suitable for 32x32 RGB images
inputs=Input(shape=(32,32,3), name='Input_Data')
```

### Convolution Layer 1
The first convolutional layer uses 32 filters with a 5x5 kernel size and 1x1 strides. This layer serves to extract general patterns and fundamental features such as edges, corners, textures, and color gradients, which form the building blocks for deeper layers to build more complex representations. In contrast to AlexNet's original design, which processes high-resolution 227x227x3 images using an 11x11 kernel with 4 strides to significantly reduce spatial dimensions and capture large patterns, our approach adapts to the smaller 32x32x3 CIFAR-10 images. To maintain efficiency without excessive downscaling, we reduced the kernel size from 11x11 to 5x5 and set a stride of 1, ensuring appropriate feature extraction tailored to the lower-resolution input.

We add batch normalization technique immidiately after each convolution layer or activation layer to normalizes activations for each mini-batch, speeding up and stabilizing training. It helps by reducing internal covariate shifts and enabling the model to learn faster with less sensitivity to weight initialization.

```python
from tensorflow.keras.layers import Conv2D, BatchNormalization

#First convolution layer with 32 filters, 5x5 kernel size, and 1x1 strides
t=Conv2D(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu', name='Conv_2D_1' )(inputs)
#First batch normalization layer
t=BatchNormalization(name='Batch_Norm_1')(t)
```

### Convolution Layer 2
The second convolutional layer builds upon the features extracted by the first layer to deepen and refine feature extraction. It captures more complex patterns and combinations of these features unique to each class, enhancing the network’s ability to recognize more complex structures within the data. By adding this layer, we increase the network's capacity to further refine the spatial context of previously learned features. To achieve this refinement, we narrow down the convolution window to a 3x3 kernel size and increase the number of filters to 96, enabling the detection of more detailed and nuanced patterns in the input data.

```python
#Second convolution layer with 96 filters, 3x3 kernel size, and 1x1 strides
t=Conv2D(filters=96, kernel_size=3, strides=1, padding="same", activation='relu', name='Conv_2D_2')(t)
#Second batch normalization layer
t=BatchNormalization(name='Batch_Norm_2')(t)
```

### Max Pooling Layer 1
The first max pooling layer is applied after the second convolutional layer to reduce the spatial dimensions of the feature maps while preserving the most prominent features crucial for accurate image recognition. As the network progresses through consecutive convolutional layers, reducing the size of the feature maps helps balance model complexity and computational efficiency, preventing an excessive number of parameters and optimizing training performance. When working with CIFAR-10 data, which consists of relatively small images, a 3x3 pooling window with a stride of 2 strikes a balance between computational efficiency and model accuracy by focusing on the strongest activations, thereby retaining key structures and patterns essential for effective feature extraction.

Dropout with a 0.4 rate was utilized as a regularization technique to combat overfitting by randomly deactivating 40% of the neurons during training. This process effectively "drops out" their contribution for each iteration, which helps prevent the model from becoming overly dependent on specific nodes. By doing so, the network is encouraged to learn more diverse, robust, and generalizable features, thereby enhancing overall model performance and minimizing the risk of overfitting.

```python
from tensorflow.keras.layers import MaxPooling2D, Dropout

#First max pooling layer with 3x3 pool size and 2x2 strides
t=MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='Max_Pool_1')(t)
#First drop out layer
t=Dropout(rate=0.4, name='Drop_Out_1')(t)
```

### Convolution Layer 3
With additional convolution layers, the CNN continues to extract and refine features that distinguish each class.

```python
#Third layer of convolution with 256 filters, 3x3 kernel size, and 1x1 strides
t=Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv_2D_3')(t)
#Third batch normalization layer
t=BatchNormalization(name='Batch_Norm_3')(t)
```

### Convolution Layer 4
We opted for only 4 convolution layers instead of AlexNet's original deeper architecture because, with smaller input images like those in CIFAR-10, fewer layers are typically sufficient to effectively extract all necessary features. This approach also helps manage computational demands, making it suitable for the limited resources available on the Google Colab platform. By reducing the number of layers, we decreased both the model size and the number of parameters.

```python
#Forth layer of convolution with 96 filters, 3x3 kernel size, and 1x1 strides
t=Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv_2D_4')(t)
#Forth batch normalization layer
t=BatchNormalization(name='Batch_Norm_4')(t)
```

### Max Pooling Layer 2

```python
#Second max pooling layer with 3x3 pool size and 2x2 strides
t=MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='Max_Pool_2')(t)
#Second drop out layer
t=Dropout(rate=0.4, name='Drop_Out_2')(t)
```

### Fully Connected (Dense) Layer 1
The Flatten layer transforms the multi-dimensional output of convolutional layers into a one-dimensional vector, enabling fully connected layers to process extracted features for classifying images.

We implemented two consecutive dense layers with 2048 and 512 neurons, respectively, after the Flatten layer, which outputs 3456 parameters, to balance model complexity and minimize the risk of overfitting. Compared to AlexNet's three dense layers with 4096, 4096, and 1000 neurons, our design with two layers of 2048 and 512 neurons is more appropriate for a compact dataset like CIFAR-10. It provides sufficient capacity to learn complex relationships and patterns from the features extracted by the convolutional layers while avoiding excessive parameters. This approach ensures computational efficiency, making the model suitable for the resource-constrained Google Colab platform while maintaining high performance.

```python
from tensorflow.keras.layers import Flatten, Dense

#Define first fully connected layer
#Flatten layer
y1=Flatten(name='Flatten_y1')(t)
#Dense layer
y1=Dense(4096, activation='relu', kernel_initializer='glorot_uniform', name='Dense_y1')(y1)
#Drop out layer
y1=Dropout(rate=0.4, name='Drop_Out_y1')(y1)
```

### Fully Connected (Dense) Layer 2

```python
#Define second fully connected layer
#Dense layer
y2=Dense(4096, activation='relu', kernel_initializer='glorot_uniform', name='Dense_y2')(y2)
#Drop out layer
y2=Dropout(rate=0.4, name='Drop_Out_y2')(y2)
```

### Output Layer
The output layer in our CNN is a dense layer consisting of 10 neurons, corresponding to the 10 classes in the dataset, with a softmax activation function. This layer converts the model’s outputs into probabilities, allowing for clear class predictions while ensuring the sum of probabilities equals 1. The softmax probabilities are used with the categorical cross-entropy loss function to calculate the model's error during training, guiding it to improve classification accuracy. This structure enables the network to make accurate, interpretable predictions for multi-class classification tasks like CIFAR-10

```python
#Define output layer
outputs=Dense(10, activation='softmax', name='Output_Model')(y2)
```

### Compile Model
```python
from tensorflow.keras.models import Model

#Define model
model=Model(inputs, outputs)
#Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
```



### Model Summary and Architecture Diagram
```
#Create a model
model=CNN_model()

#Print model summary
model.summary()
print("\n\n\n")

#Plot model architecture
keras.utils.plot_model(model=model,show_shapes=True, show_layer_names=True)
```


## Model Training

### Monitoring


```python

#Initialize Tensor Board - logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

```

* Early Stopping: This method monitors the model's performance on a validation data set to halt training once performance begins to deteriorate, preventing it from overfitting and memorizing noise in the training data.


```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Define callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1)
earlyStopping_callback = EarlyStopping(monitor='val_accuracy', patience=5)
```



```python
#Train the model on the trainig data set using 32 batch size, 150 epochs,0.1 validation split, and early stopping call back
history=model.fit(x_train, y_train, batch_size=32, epochs=150, validation_split=0.1, callbacks=[tensorboard_callback, earlyStopping_callback], verbose=2)

```

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


