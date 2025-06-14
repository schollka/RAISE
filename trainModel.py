import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Daten laden ===
X = np.load("X.npy")  # ggf. Pfad anpassen
y = np.load("y.npy")

print(f"Daten geladen: X = {X.shape}, y = {y.shape}")

# === 2. Split in Training und Validierung ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Trainingsdaten: {X_train.shape}, Validierungsdaten: {X_val.shape}")

# === 3. Modell definieren ===
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# === 4. Training ===
earlyStop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[earlyStop]
)

# === 5. Modell speichern ===
model.save("landingClassifier.h5")
print("Modell gespeichert als landingClassifier.h5")

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
