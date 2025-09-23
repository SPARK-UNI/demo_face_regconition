# train_cnn_color.py
import os, cv2, numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

DATASET_PATH = "Face_data_aug"   # load dataset sau khi augment
IMG_SIZE = (96, 96)

# ===== Load dataset =====
def load_dataset(dataset_path, img_size=(96,96)):
    X, y, labels = [], [], []
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir): 
            continue
        if class_name not in labels: 
            labels.append(class_name)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None: 
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(class_name)
    labels = sorted(labels)
    label_to_idx = {name:i for i,name in enumerate(labels)}
    y = np.array([label_to_idx[v] for v in y])
    return np.array(X), y, labels

X, y, label_names = load_dataset(DATASET_PATH, IMG_SIZE)
X = X.astype("float32") / 255.0
y = to_categorical(y, num_classes=len(label_names))

# Train/val split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===== Model CNN =====
model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(255, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(label_names), activation="softmax"))

model.summary()

# ===== Compile & Train =====
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ===== Evaluate =====
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# ===== Save model =====
model.save("face_cnn_model.h5")
with open("labels.txt", "w", encoding="utf-8") as f:
    for i, name in enumerate(label_names):
        f.write(f"{i} {name}\\n")
