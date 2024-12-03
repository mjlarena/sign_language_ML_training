# CNN Training to detect sign language patterns 


## About the project
This project was developed as the final project for an ML professional certificate, where we could practice important things we had learned from the program. The project consists of training a convolutional neural network to categorize images into different options from sign language gestures. It was trained testing two different CNN model architectures and then a Hyperparameter tuning process was performed to get the best parameters for the model. During this Hyperparameter tunning process, a Bayesian optimization process was applied and run for 100 iterations to get the result after testing different combinations for the Hyperparameters. Finally, The model was trained with those parameters, reaching a 92% accuracy on the test set, meaning that the model accurately categorized 92% of the pictures from the test set.


## DATA
The data was obtained from a dataset available at Kaggle:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

![Screenshot](amer_sign2.png)

The Sign Language MNIST dataset is a collection of 28x28 grayscale images representing hand gestures for 24 letters of the American Sign Language (ASL) alphabet. The samples exclude the letters "J" and "Z," since they have movements. The dataset is divided into 27,455 images in the training set and 7,172 images for the testing set. Each image has its own label corresponding to the letters from the ASL gesture.

The dataset is preprocessed, with all images resized and normalized and it is publicly available on Kaggle and intended for educational and research purposes. As mentioned before, this dataset does not cover dynamic gestures which would be crucial to complete the ASL interpretation in full.

## MODEL 
The model used for this project was a Convolutional Neural Network (CNN) because it can perform well for image recognition thanks to its architecture. Two different architectures were tested:

First architecture:
- 1 input layer for the image in 1 channel
- First convolutional layer (3x3 Kernel size)
- ReLU activation function to introduce non-linearity
- Pooling Layer (2x2)
- Second convolutional Layer
- Second ReLU activation function
- Pooling Layer (2x2)
- Flattening of data, converting it to a 1D feature vector.
- 2 fully connected layers, the last one outputs a vector of size equal to the number of classes.

After testing the first architecture I was able to see a performance of 100% of accuracy for the validation set and 86,5% for the test set, which in general terms is not bad but shows a high level of overfitting, so given the problem and the dataset, it was likely that another model could work better. Therefore, I added a dropout layer to reduce overfitting.

Second architecture:
- It adds dropout layers to reduce the overfitting.
- Additional fully connected layer

After seeing an improvement using this new architecture, I tunned the Hyperparameters using a Bayesian Optimization process.

## HYPERPARAMETER OPTIMIZATION

The Hyperparameter tuning process focused on optimizing the Hyperparameters of the Convolutional Neural Network (CNN) architecture defined above.

The hyperparameters tuned for the CNN architecture were:

- conv1_filters (16–64) and conv2_filters (32–128): Number of filters in the convolutional layers for feature extraction
- kernel_size (3–5): Size of convolutional kernels
- pool_kernel_size (2–3): Max-pooling kernel size for spatial dimension reduction
- fc1_units (64–512) and fc2_units (32–256): Neurons in the fully connected layers
- dropout_rate (0.2–0.5): Dropout rate to reduce overfitting

Optimization strategy:
I used Bayesian optimization with Gaussian Processes to efficiently explore the hyperparameter space defined above. The objective function evaluated validation accuracy after 10 training epochs each time. The best parameters were selected after 100 evaluations, ensuring an optimal balance between computational efficiency and model performance. The code was run using a Google Colab with an activated GPU.


## RESULTS
A summary of your results and what you can learn from your model 

You can include images of plots using the code below:
![Screenshot](image.png)

## (OPTIONAL: CONTACT DETAILS)
If you are planning on making your github repo public you may wish to include some contact information such as a link to your twitter or an email address. 
