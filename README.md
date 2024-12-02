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
A summary of the model you’re using and why you chose it. 

## HYPERPARAMETER OPTIMSATION
Description of which hyperparameters you have and how you chose to optimise them. 

## RESULTS
A summary of your results and what you can learn from your model 

You can include images of plots using the code below:
![Screenshot](image.png)

## (OPTIONAL: CONTACT DETAILS)
If you are planning on making your github repo public you may wish to include some contact information such as a link to your twitter or an email address. 
