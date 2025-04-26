#Exercise 3:
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import cv2

# Define paths to the dataset
train_dir = 'train'
validation_dir = 'validation'

# Image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Only rescaling for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary') # Binary classification: cats (0) vs dogs (1)

# Load and preprocess validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary')

# Build the CNN model
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
MaxPooling2D(2, 2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Conv2D(128, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Flatten(),
Dense(512, activation='relu'),
Dropout(0.5),
Dense(1, activation='sigmoid')]) # Binary output

# Compile the model
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
EPOCHS = 5
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Save the model
model.save('cats_vs_dogs_model.h5')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_history.png')
plt.show()

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
    return img

cat_img = cv2.imread('cat.png')
cat_img_preprocessed = preprocess_img(cat_img)
dog_img = cv2.imread('dog.png')
dog_img_preprocessed = preprocess_img(dog_img)

cat_pred = model.predict(cat_img_preprocessed)
cat_pred = 'cat' if cat_pred[0][0] < 0.5 else 'dog'
dog_pred = model.predict(dog_img_preprocessed)
dog_pred = 'cat' if dog_pred[0][0] < 0.5 else 'dog'

plt.imshow(cat_img)
plt.title(f'Prediction: {cat_pred}')
plt.axis('off')
plt.show()

plt.imshow(dog_img)
plt.title(f'Prediction: {dog_pred}')
plt.axis('off')
plt.show()