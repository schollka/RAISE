'''
              __|__
       --------(_)--------       
              RAISE         
 Runway Approach Identification for Silent Entries
------------------------------------------------------
    Tracking • Prediction • Silent Entry Detection     
------------------------------------------------------

model training script
'''

print("Loading modules.")

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, MaxPooling1D, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml


print("Modules loaded")


def findOptimalThresholdF2(yTrue, yPredProb, beta=2.0, verbose=True):
    """
    Returns the threshold that maximizes the F-beta score (default beta=2 for recall emphasis).
    
    Parameters:
        yTrue (array): True binary labels
        yPredProb (array): Predicted probabilities from model
        beta (float): Weight of recall in F-beta score
        verbose (bool): If True, prints best threshold and score
        
    Returns:
        bestThreshold (float): Threshold with highest F-beta score
        bestFbeta (float): Corresponding F-beta score
    """
    thresholds = np.linspace(0.1, 0.9, 81)  # steps of 0.01
    bestThreshold = 0.5
    bestFbeta = -1.0

    for t in thresholds:
        yPred = (yPredProb > t).astype(int)
        fbeta = fbeta_score(yTrue, yPred, beta=beta)
        if fbeta > bestFbeta:
            bestFbeta = fbeta
            bestThreshold = t

    if verbose:
        print(f"Optimal threshold (F{beta:.1f}-score): {bestThreshold:.2f}")
        print(f"Best F{beta:.1f}-score: {bestFbeta:.4f}")

    return bestThreshold, bestFbeta


sourceCodeDir = os.path.dirname(os.path.abspath(__file__))
parameterFile = os.path.join(sourceCodeDir, "parameters.yaml")

# Load parameters
print("Loading parameters.")
with open(parameterFile, "r") as file:
    allParams = yaml.safe_load(file)

databaseParameters = allParams["databaseParameters"]
machineLearningParameters = allParams["machineLearningParameters"]

outputDir = os.path.dirname(os.path.abspath(databaseParameters["DATABASE_PATH"]))

print("Parameters loaded")

#load X and Y
print("Loading training data")
X = np.load(os.path.join(outputDir, "X.npy"))
y = np.load(os.path.join(outputDir, "y.npy"))

print(f"Data loaded: X = {X.shape}, y = {y.shape}")

#split data into training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data: {X_train.shape}, validation data: {X_val.shape}")

#define model (input includes 6 features: relativeTime + 5 flight features)
model = Sequential([
    # first convolutional layer with a wider kernel to capture longer patterns (e.g. descent trends)
    Conv1D(64, 7, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.2),

    # max pooling reduces the temporal dimension and highlights strong activations
    MaxPooling1D(pool_size=2),

    # second convolutional layer with a smaller kernel to capture finer local features
    Conv1D(32, 5, activation='relu', padding='same'),
    Dropout(0.2),

    # third convolutional layer for additional abstraction
    Conv1D(16, 3, activation='relu', padding='same'),
    Dropout(0.1),

    # global pooling aggregates features across the time axis
    GlobalAveragePooling1D(),

    # dense layer for final decision logic
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),

    # output layer for binary classification (landing vs not landing)
    Dense(1, activation='sigmoid')
])


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

#train model
print("Start training model.")
class_weights = {0: 1.0, 1: float(len(y) - sum(y)) / float(sum(y))}
earlyStop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[earlyStop],
    class_weight=class_weights
)

#save model
model.save(os.path.join(outputDir, "landingClassifier.keras"))
print("Modell saved")

#converting into tflite
print("Converting model into tflite model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tfliteModel = converter.convert()

with open(os.path.join(outputDir, "landingClassifierLite.tflite"), "wb") as f:
    f.write(tfliteModel)

print("TFLite model saved.")

#evaluation
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_acc:.4f}")

#visualization Loss & Accuracy
# plot and save accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Verlauf')
plt.tight_layout()
plt.savefig(os.path.join(outputDir, 'accuracy_plot.png'))  # save to outputDir
plt.close()

# plot and save loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Verlauf')
plt.tight_layout()
plt.savefig(os.path.join(outputDir, 'loss_plot.png'))  # save to outputDir
plt.close()

y_pred_prob = model.predict(X_val, verbose=0)

# predict probabilities on validation set
y_pred_prob = model.predict(X_val, verbose=0)

# find optimal threshold
bestThreshold, bestF2 = findOptimalThresholdF2(y_val, y_pred_prob, beta=1.5)

# evaluate final classification
y_pred = (y_pred_prob > bestThreshold).astype(int)

reportPath = os.path.join(outputDir, "classification_report.txt")
with open(reportPath, "w") as f:
    # write optimal threshold report
    f.write(f"Classification report with optimal threshold ({bestThreshold:.2f}):\n")
    f.write(classification_report(y_val, y_pred))
    f.write("\n\n")

    # write threshold analysis for multiple values
    for t in np.arange(0.4, 0.8, 0.05):
        y_pred_threshold = (y_pred_prob > t).astype("int32")
        f.write(f"Threshold: {t:.2f}\n")
        f.write(classification_report(y_val, y_pred_threshold, digits=3))
        f.write("\n\n")

# print to console as before
print(f"\nClassification report with optimal threshold ({bestThreshold:.2f}):")
print(classification_report(y_val, y_pred))

for t in np.arange(0.4, 0.8, 0.05):
    y_pred = (y_pred_prob > t).astype("int32")
    print(f"\nThreshold: {t:.2f}")
    print(classification_report(y_val, y_pred, digits=3))
