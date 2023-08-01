import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import optuna

# Load the dataset (assuming you have 'train.csv' and 'test.csv' in the same directory)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data preprocessing and feature selection
X = train_data.drop('label', axis=1)
y = train_data['label']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape and normalize the data
X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0
X_val = X_val.values.reshape(-1, 28, 28, 1) / 255.0
X_test = test_data.values.reshape(-1, 28, 28, 1) / 255.0

# Define the objective function for Optuna
def objective(trial):
    # Parameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dense_units = trial.suggest_int('dense_units', 32, 256)

    # Build the deep learning model
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model with the suggested learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with early stopping
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[es_callback], verbose=0)

    # Evaluate the model on the validation set
    score = model.evaluate(X_val, y_val, verbose=0)
    return score[1]

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters from Optuna
best_params = study.best_params

# Build the final model with the best hyperparameters
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(best_params['dense_units'], activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the final model with the best learning rate
optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the final model with early stopping
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[es_callback])

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Save the predictions to a CSV file for submission
submission = pd.DataFrame({'ImageId': range(1, len(y_pred_labels) + 1), 'Label': y_pred_labels})
submission.to_csv('submission.csv', index=False)
