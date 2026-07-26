import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Data Training
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
    [6.0, 2.2, 4.0, 1.0],
    [5.5, 2.3, 4.0, 1.3],
], dtype=np.float32)

y_train = np.array([
    0,
    0,
    2,
    2,
    1,
    1
], dtype=np.int32)

# Data Testing
X_test = np.array([
    [5.0, 3.4, 1.5, 0.2],
    [6.1, 2.8, 4.7, 1.2],
    [6.5, 3.0, 5.5, 1.8]
], dtype=np.float32)

y_test = np.array([
    0,
    1,
    2
], dtype=np.int32)

# Model ANN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    verbose=0
)

# Evaluasi
loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("=" * 60)
print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Test Loss     : {loss:.4f}")
print("=" * 60)

# Prediksi
output = model.predict(X_test, verbose=0)

print("\nOutput Mentah (Softmax)")
print(output)

prediksi = np.argmax(output, axis=1)

print("\nHasil Prediksi")
print("=" * 60)

for i in range(len(X_test)):
    print(f"Data ke-{i+1}")
    print(f"Input          : {X_test[i]}")
    print(f"Output Softmax : {output[i]}")
    print(f"Prediksi       : Kelas {prediksi[i]}")
    print(f"Label Asli     : Kelas {y_test[i]}")

    if prediksi[i] == y_test[i]:
        print("Status         : Benar")
    else:
        print("Status         : Salah")

    print("-" * 60)

# Grafik Accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"], label="Accuracy", linewidth=2)
plt.plot(history.history["loss"], label="Loss", linewidth=2)
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()