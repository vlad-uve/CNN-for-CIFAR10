# Implementation of CNN to Classify CIFAR-10 Images
## Project Overview

This project was created for the course "YCBS 258 - Practical Machine Learning" at Mcgill University. This coure aimed implementation of a CNN model for classification of images from CIFAR-10 data set in Google Collab environment. Target model accuracy to achieve was 0.8. AlexNet's architecture was chosen as a reference for an architecture of the model to improve accuracy, but with certain simplifications due to the platform limitations and the problem scope. In addition, certain regularization techniques were incorporated to  



## Data Used





kernel size is 5x5 because the input is 32x32

## Model Created

## Results



 #Choosing not to incorporate a pooling layer after the initial convolution layer 


regularization techniques

#Above model was inspired by Alexnet but several changes were put in place in order to improve accuracy.
#The changes were aimed at reducing complexity and also considering the initial data size - and how this flowed through the model
#Added batch normalization and dropouts in several points throughout the model to help improve results.


# Assignment
Build the AlexNet architecture and train it on the CIFAR10 dataset.

You may resize some of the kernel sizes (mainly the first one) and the number of kernels because CIFAR100 is a smaller dataset than ImageNet, as well as lower-resolution (32x32 vs 224x224).

You may resize the number of kernels used per layer if it speeds up training and/or preserves accuracy. The exact AlexNet architecture and number of units will not work.

Report training, validation and test errors. Which hyperparameters did you change to improve performance?

**MAKE SURE YOU USE A GPU!** Top menu select "Runtime", then "Runtime Type". Make sure "Hardware Accelerator" has the option "GPU" selected.

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


