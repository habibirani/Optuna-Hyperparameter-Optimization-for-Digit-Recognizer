# Optuna Hyperparameter Optimization for Digit Recognizer

This repository contains a Python script for optimizing the hyperparameters of a Convolutional Neural Network (CNN) model using Optuna, a hyperparameter optimization library. The model is trained on the famous "Digit Recognizer" dataset from Kaggle, which involves classifying handwritten digits (0 to 9).

## Dataset

The dataset consists of grayscale images of size 28x28 pixels, representing handwritten digits. The train.csv file contains labeled data with pixel values and corresponding digit labels, while the test.csv file contains the pixel values of unlabeled images used for predictions.

## Dependencies

The code uses the following Python libraries:

- pandas
- numpy
- scikit-learn
- TensorFlow
- Optuna

## Usage

1. Make sure you have the required datasets "train.csv" and "test.csv" in the same directory as the script.
2. Install the necessary dependencies using `pip install pandas numpy scikit-learn tensorflow optuna`.
3. Run the Python script "digit_recognizer_optuna.py".

The script will perform hyperparameter optimization using Optuna and train the CNN model with the best hyperparameters found. It will save the predictions on the test set in "submission.csv" for submission to the Kaggle competition.

## Hyperparameter Optimization

The objective function for Optuna is defined to optimize the learning rate and the number of dense units in the fully connected layer of the CNN. The early stopping callback is used to prevent overfitting during model training.

The number of trials for the optimization process is set to 50, which means Optuna will explore 50 different sets of hyperparameters to find the best combination.

## Results

After running the optimization process, the script will display the best hyperparameters found by Optuna. The final model will be trained with these hyperparameters, and the test set predictions will be saved to "submission.csv".

Feel free to experiment with the number of trials, learning rate range, and dense units range to further improve the model's performance.
