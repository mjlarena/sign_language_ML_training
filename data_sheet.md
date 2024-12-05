# Datasheet

URL: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

## Motivation

The dataset was created to enable machine learning research and applications for recognizing American Sign Language (ASL) gestures represented by letters (A-Z). It can serve as a training and evaluation benchmark for computer vision models on gesture recognition tasks.
  
The dataset was uploaded to Kaggle by a profile named Tecperson, but it is not clear who that person is. The original MNIST dataset framework it draws inspiration from was funded by various academic and research initiatives, but the funding and specific creators of this dataset are unclear from the provided details.
 
## Composition

The instances represent grayscale images of hand gestures corresponding to the 24 letters of the ASL alphabet (excluding J and Z due to their dynamic movements). The information is in two CSV files.

It contains the following information:
- Training set: 27,455 images.
- Testing set: 7,172 images.

Each image is a 28x28 pixel grayscale representation of a hand gesture and there is no indication of missing data in the dataset. The dataset does not include confidential data. The images are cropped, grayscale representations of hand gestures, with no identifiable personal or sensitive information.


## Collection process

The original dataset comprised 1,704 color images of hand gestures from multiple users performing gestures against various backgrounds. These images were processed and greatly extended using an automated pipeline to create over 34,000 images. The process included cropping to the hands-only region, gray-scaling, resizing, and applying a variety of augmentation techniques to expand the dataset.


## Preprocessing/cleaning/labelling

The dataset available at Kaggle had already been significantly preprocessed and, for this task, no further preprocessing was needed. When reviewing the preprocessing that was already performed on the data, we can see the following:
- Cropping images to focus on the hands-only region.
- Converting to grayscale.
- Resizing to 28x28 pixels.
Applying augmentations such as:
- Filters ('Mitchell', 'Robidoux', 'Catrom', 'Spline', 'Hermite').
- 5% random pixelation.
- ±15% brightness/contrast adjustments.
- 3 degrees rotations.

These techniques increased the dataset size and improved its variability for model training.

 
## Uses

The dataset can be used for tasks such as:
- Gesture recognition model training and evaluation.
- Implementation of ASL recognition tasks (not including dynamic gestures).

Since the dataset does not include dynamic gestures (like "J" or "Z"), it may not generalize well to real-life ASL interpretation without additional datasets. The dataset should not be used as a standalone benchmark for dynamic gesture recognition or full ASL interpretation, as it only includes static gestures for 24 letters.


## Distribution

The dataset is publicly available on Kaggle and can be downloaded by anyone with a Kaggle account. The dataset is hosted under Kaggle's terms of service, which permits its use for research and educational purposes. The license for the dataset is CC0: Public Domain.


## Maintenance

The dataset is hosted and maintained on Kaggle by Tecperson. Further maintenance or updates are unclear, but it was updated in 2017 (7 years ago) as stated on Kaggle's site.

