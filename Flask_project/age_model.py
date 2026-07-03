import os
import zipfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

zip_path="/content/archive (2).zip"

extract_path="/content/age_dataset"

with zipfile.ZipFile(zip_path,'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset Extracted Successfully")

csv_path="/content/age_dataset/faces/train.csv"

df=pd.read_csv(csv_path)

df.head()

image_folder="/content/age_dataset/faces/Train"

df["image"]=df["ID"].apply(lambda x: os.path.join(image_folder,x))

df["label"]=df["Class"].str.lower()

df=df[["image","label"]]

df.head()

print(df.shape)

print(df["label"].value_counts())

df.sample(5)

image_paths = df['image'].tolist()
labels = df['label'].tolist()

print("Total Images:", len(image_paths))
print("Example Image Path:", image_paths[0] if image_paths else "No images found")
print("Example Label:", labels[0] if labels else "No labels found")

df = pd.DataFrame({
    "image": image_paths,
    "label": labels
})

df.head()

plt.figure(figsize=(12,6))

for i in range(6):

    plt.subplot(2,3,i+1)

    index = random.randint(0,len(df)-1)

    img = Image.open(df.iloc[index]["image"])

    plt.imshow(img)

    plt.title(df.iloc[index]["label"])

    plt.axis("off")

plt.show()

label_encoder = LabelEncoder()

df["encoded_label"] = label_encoder.fit_transform(df["label"])

df.head()

import pickle

with open("label_encoder.pkl","wb") as f:
    pickle.dump(label_encoder,f)

print("Label Encoder Saved!")

y = to_categorical(df["encoded_label"])

print(y.shape)

train_df, test_df = train_test_split(

    df,

    test_size=0.20,

    stratify=df["label"],

    random_state=42

)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,

    rotation_range=20,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True

)

test_datagen = ImageDataGenerator(

    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input

)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image",
    y_col="label",
    target_size=(128,128),
    batch_size=64,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image",
    y_col="label",
    target_size=(128,128),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(128,128,3)

)

base_model.trainable = False

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(512,activation="relu")(x)

x = Dropout(0.5)(x)

output = Dense(

    len(label_encoder.classes_),

    activation="softmax"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=output

)

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

early = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3

)

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(

    "best_model.keras",

    save_best_only=True

)

history = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=10,

    callbacks=[early,reduce,checkpoint]

)

loss,accuracy = model.evaluate(test_generator)

print("Accuracy :",accuracy)

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.legend(["Train","Validation"])

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.legend(["Train","Validation"])

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()

prediction = model.predict(test_generator)

predicted = np.argmax(prediction,axis=1)

from sklearn.metrics import classification_report

print(classification_report(

    test_generator.classes,

    predicted

))

model.save("age_model.keras")

from tensorflow.keras.models import load_model

model = load_model("age_model.keras")