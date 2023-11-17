# Face Mask Detection using Convolutional Neural Network (CNN)

This project aims to build a Face mask detection system using Convolutional Neural Networks (CNNs). The system can automatically detect whether a person is wearing a mask or not based on an input image.

## Dataset

The dataset used for this project is obtained from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset), which contains images of people wearing and not wearing masks. The images are labeled with 1 for people wearing masks and 0 for people not wearing masks.

## Preprocessing

The images are processed using various libraries such as OS, NumPy, Matplotlib, OpenCV, and Pillow. The images are converted to numpy arrays and resized to a standard size of 128x128x3. The dataset is then split into training and testing sets, with 80% for training and 20% for testing. The pixel values of the images are scaled down to a range of 0 to 1.

## Model Architecture

The neural network model is built using TensorFlow and Keras libraries. The model architecture consists of convolutional layers, max pooling layers, dense layers, and a dropout layer to prevent overfitting. The output layer has two neurons, one for each class in the dataset (mask and no mask). The model is compiled with an optimization algorithm and a loss function.

## Model Training and Evaluation

The model is trained using the training dataset and validated using a validation split. The performance of the model is evaluated based on loss and accuracy metrics. The trained model is then evaluated on the test dataset to assess its performance.

## Predictive System

A predictive system is built using the trained model. A single image is reshaped and fed into the model for prediction. The system outputs whether the person in the image is wearing a mask or not.

## Deployment Options

The mask detection system can be deployed using various options such as creating a user interface (UI) or an application programming interface (API). This allows the system to be integrated into different platforms or systems for real-time mask detection.

## Conclusion

In conclusion, this project demonstrates the process of building a face mask detection system using CNNs. The system can effectively detect whether a person is wearing a mask or not based on input images. The deployment options provide flexibility in integrating the system into different applications or platforms to ensure compliance with mask mandates and prevent the spread of COVID-19.

#### To execute the project, start by downloading the Kaggle JSON file(Kaggke's Beta API). Next, click/open the Google Colab link provided in the bio. Upload the Kaggle.json file to Colab, and then run the project.
