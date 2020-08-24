# Optical Character Recognition

Optical character recognition with SVM and CNN. The goal of the project is to detect and classify images of letters. 

## Overview
The system consists of four distinct parts; preprocessing, SVM, CNN and sliding-window character detection and classification.  
The SVM and CNN classifiers are independent of each other, but both require specific preprocessing of data, 
while the character detection and classification is dependent on the CNN model to function properly. 

## How to run

### Requirements
To run the code you will need the following libraries installed on your system:
* Numpy - For handling images as arrays
* Keras & Tensorflow - For the CNN. If you have an old version of Keras or Tensorflow you may need to
update to a newer version, namely 2.3.1 and 2.0.0, respectively.
* scikit-learn & scikit-image - For the SVM & for splitting the dataset and simple data-processing.
* Pillow - For loading images in the dataset
* Matplotlib - For plotting Accuracy and Loss of the CNN after training

### Run the code
The project can be run by navigating to the project root-folder and executing the file `program.py`.   
The CNN model is saved in the .h5-file, which is used for the detection-classification task.  
In program.py you can set the train svm and train cnn variables if you want to train the networks yourself.  
Note that you have to train the SVM in order to run it.




