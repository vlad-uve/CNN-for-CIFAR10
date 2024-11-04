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
