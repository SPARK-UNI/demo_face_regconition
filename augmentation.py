# augmentation.py
import os, cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn dataset gốc & dataset augment
SRC_DIR = "Face_data/data_mau"
DST_DIR = "Face_data_aug"
IMG_SIZE = (96, 96)
AUG_PER_IMAGE = 5   # số ảnh augment mỗi ảnh gốc

# Cấu hình augmentation
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

os.makedirs(DST_DIR, exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    src_class = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(src_class): 
        continue
    dst_class = os.path.join(DST_DIR, class_name)
    os.makedirs(dst_class, exist_ok=True)

    for fname in os.listdir(src_class):
        fpath = os.path.join(src_class, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)

        # Lưu ảnh gốc
        base_name = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(dst_class, base_name + "_orig.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Sinh augment
        x = img[np.newaxis, ...]
        aug_iter = datagen.flow(x, batch_size=1)
        for i in range(AUG_PER_IMAGE):
            aug_img = next(aug_iter)[0].astype(np.uint8)
            out_path = os.path.join(dst_class, f"{base_name}_aug{i}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

print("Augmentation done. Saved to:", DST_DIR)
