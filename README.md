##This project was created for the course "YCBS 258 - Practical Machine Learning" at Mcgill University

Alexnet architecture was chosen as a reference for the architecture of the model, but significantly simplified

Task scope implies simplifications as well

alexnet is aimed improve accuracy

Alexnet architecture was chosen as a reference for the architecture of the model, but significantly simplified due to the platform limitations and task scope

kernel size is 5x5 because the input is 32x32


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
