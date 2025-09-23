import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

def load_dataset(dataset_path):
    X, y = [], []
    label_names = []
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        # Add person name to labels if not already added
        if person_name not in label_names:
            label_names.append(person_name)
        
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (60, 60)).flatten()
            X.append(img)
            y.append(person_name)
    
    return np.array(X), np.array(y), sorted(label_names)

# Đường dẫn dataset
dataset_path = 'Face_data/data_mono'

# Load dữ liệu
X, y, label_names = load_dataset(dataset_path)

# Manual label encoding (không dùng sklearn)
label_to_index = {name: idx for idx, name in enumerate(label_names)}
y_encoded = np.array([label_to_index[name] for name in y])
num_classes = len(label_names)

print(f"Found {len(label_names)} classes: {label_names}")
print(f"Total samples: {len(X)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Chuẩn hóa dữ liệu về [0,1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot encoding cho y
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Xây dựng ANN thuần túy - chỉ Dense layers
model = Sequential([
    Input(shape=(60*60,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
print("\nModel Architecture:")
model.summary()

# Early stopping để tránh overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train model
print("\nStarting training...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Đánh giá
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f'\nFinal Test Accuracy: {acc:.4f}')
print(f'Final Test Loss: {loss:.4f}')

# Lưu model
model.save("keras_model.h5")
print("Model saved as: keras_model.h5")

# Tạo file labels.txt
with open("labels.txt", "w", encoding="utf-8") as f:
    for idx, name in enumerate(label_names):
        f.write(f"{idx} {name}\n")

print("Labels saved as: labels.txt")
print(f"Labels: {label_names}")

# In thông tin training history
print(f"\nTraining completed in {len(history.history['loss'])} epochs")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")