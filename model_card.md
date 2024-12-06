# Model Card

## Model Description

**Input:** The inputs of the model are grayscale images of size 28x28 pixels. These images represent hand gestures corresponding to the letters of the American Sign Language (ASL) alphabet.

**Output:** The model classifies input images into 25 categories. Each category corresponds to a different letter in the ASL alphabet, excluding the letters 'J' and 'Z', which require motion to be expressed.

**Model Architecture:**
The model is a Convolutional Neural Network (CNN) with the following layers:

Convolutional Layer 1: 25 filters, 3x3 kernel, ReLU activation, followed by max-pooling (3x3).
Convolutional Layer 2: 40 filters, 3x3 kernel, ReLU activation, followed by max-pooling (3x3).
Dropout Layer: A dropout rate of 30% to reduce overfitting.
Fully Connected Layer 1: 156 neurons with ReLU activation.
Fully Connected Layer 2: 32 neurons with ReLU activation.
Output Layer: 25 neurons with no activation function (logits). The architecture was fine-tuned using hyperparameter optimization to determine the best number of filters, kernel size, and fully connected units.

## Performance

After tuning the hyperparameters of the model, it shows a performance of 95,9% on the test set, which makes it a good model to categorize hand gesture images into categories of the American Sign Language and shows that it generalizes well for unseen information. 

## Limitations

This model can not recognize or classify the gestures for 'J' and 'Z' as they have motion. This model was only trained using still images and therefore an additional model should be needed in case of wanting to have all the letters of the ASL.

## Trade-offs

The hyperparameters may be tuned specifically for a particular dataset as stated before. While this results in optimal performance on that dataset, the model may not generalize as well to other similar datasets or real-life images unless retrained or fine-tuned.

During the hyperparameter tuning process, there is an exploration-exploitation trade-off. The model might be optimized for a narrow set of parameters that work best within the known search space (defined in the tunning process). While this shows high performance and accuracy for the current dataset, it may limit the exploration of other possible configurations (maybe outside of the defined space) that could improve the model’s flexibility or robustness.
