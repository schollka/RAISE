print("start")

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

print("Import fertig")

sourceCodeDir = os.path.dirname(os.path.abspath(__file__))
parameterFile = os.path.join(sourceCodeDir, "parameters.yaml")

# Load parameters
with open(parameterFile, "r") as file:
    allParams = yaml.safe_load(file)

databaseParameters = allParams["databaseParameters"]
machineLearningParameters = allParams["machineLearningParameters"]

outputDir = os.path.dirname(os.path.abspath(databaseParameters["DATABASE_PATH"]))

print("parameters geladen")

# === X und y laden ===
X = np.load(os.path.join(outputDir, "X.npy"))
y = np.load(os.path.join(outputDir, "y.npy"))

print(f"Daten geladen: X = {X.shape}, y = {y.shape}")

# === 2. Split in Training und Validierung ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Trainingsdaten: {X_train.shape}, Validierungsdaten: {X_val.shape}")

# === 3. Modell definieren ===
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

# === 4. Training ===
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

# === 5. Modell speichern ===
model.save("landingClassifier.keras")
print("Modell gespeichert")

# === 5a. Konvertierung in TFLite (nur BUILTIN Ops) ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tfliteModel = converter.convert()

with open("landingClassifier_lite_compatible.tflite", "wb") as f:
    f.write(tfliteModel)

print("TFLite-kompatibles Modell erfolgreich gespeichert.")

# === 6. Evaluation ===
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validierungsgenauigkeit: {val_acc:.4f}")

# === 7. Visualisierung: Loss & Accuracy ===
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

# === 8. Confusion Matrix & Klassifikationsbericht ===
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\nKlassifikationsbericht:")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Vorhergesagt")
plt.ylabel("Wahr")
plt.title("Confusion Matrix")
plt.show()
