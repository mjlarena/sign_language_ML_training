# Model Card

## Model Description

**Input:** The inputs of the model are grayscale images of size 28x28 pixels. These images represent hand gestures corresponding to the letters of the American Sign Language (ASL) alphabet. The pixel values are normalized to the range [0, 1].

**Output:** The model classifies input images into 25 categories. Each category corresponds to a different letter in the ASL alphabet, excluding the letters 'J' and 'Z', which require motion to be expressed.

**Model Architecture:**
The model classifies input images into one of 25 categories. Each category corresponds to a different letter in the ASL alphabet, excluding the letter 'J' and 'Z', which require motion to be expressed.

Model Architecture:
The model is a Convolutional Neural Network (CNN) with the following layers:

Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation, followed by max-pooling (2x2).
Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation, followed by max-pooling (2x2).
Dropout Layer: A dropout rate of 40% to reduce overfitting.
Fully Connected Layer 1: 128 neurons with ReLU activation.
Fully Connected Layer 2: 60 neurons with ReLU activation.
Output Layer: 25 neurons with no activation function (logits), used in combination with softmax during evaluation to classify into one of the 25 categories.
The architecture was fine-tuned using hyperparameter optimization to determine the best number of filters, kernel size, and fully connected units.

## Performance

Give a summary graph or metrics of how the model performs. Remember to include how you are measuring the performance and what data you analysed it on. 

## Limitations

Outline the limitations of your model.

## Trade-offs

Outline any trade-offs of your model, such as any circumstances where the model exhibits performance issues. 
