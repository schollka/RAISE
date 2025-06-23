'''
              __|__
       --@--@--(_)--@--@--       
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
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

print("Modules loaded")

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

#define model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
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
model.save("landingClassifier.keras")
print("Modell saved")

#converting into tflite
print("Converting model into tflite model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tfliteModel = converter.convert()

with open("landingClassifierLite.tflite", "wb") as f:
    f.write(tfliteModel)

print("TFLite model saved.")

#evaluation
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validierungsgenauigkeit: {val_acc:.4f}")

#visualization Loss & Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Verlauf')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Verlauf')
plt.show()

#Confusion Matrix & classification report
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nClassification report:")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
