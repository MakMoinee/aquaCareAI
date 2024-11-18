import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import matplotlib.pyplot as plt

# Define paths to your dataset
image_dir = './train/images'
label_dir = './train/labels'
img_size = (299, 299)  # Xception expects 299x299 images
batch_size = 32

# Function to parse labels from YOLO format
def parse_label_from_yolo(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # Assume binary classification: 0 for 'healthy', 1 for 'disease'
    for line in lines:
        class_id = int(line.split()[0])  # The first number in each line is the class ID
        if class_id == 1:  # If any object is classified as 'disease', label the image as diseased
            return 1
    return 0  # Otherwise, label as healthy

# Function to create dataset
def create_dataset(image_dir, label_dir, subset):
    image_paths = sorted([os.path.join(image_dir, subset, fname) for fname in os.listdir(os.path.join(image_dir, subset)) if fname.endswith('.jpg')])
    label_paths = sorted([os.path.join(label_dir, subset, fname.replace('.jpg', '.txt')) for fname in os.listdir(os.path.join(image_dir, subset)) if fname.endswith('.jpg')])

    images = []
    labels = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(img_path)), img_size)
        label = parse_label_from_yolo(lbl_path)
        images.append(img)
        labels.append(label)

    images = tf.stack(images)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create training and validation datasets
train_dataset = create_dataset(image_dir, label_dir, 'train')
val_dataset = create_dataset(image_dir, label_dir, 'val')

# Load the Xception model with pre-trained ImageNet weights
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers for white spot disease classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Use a single neuron with sigmoid for binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Unfreeze the base model and fine-tune
for layer in base_model.layers:
    layer.trainable = True

# Re-compile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
fine_tune_epochs = 20
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

# Save the model in the SavedModel format
model_save_path = 'result2.h5'
model.save(model_save_path, save_format='h5')

print(f"Model saved to {model_save_path}")

# Plot training and validation loss/accuracy
def plot_training_history(history, fine_tuning_history=None):
    plt.figure(figsize=(14, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Initial Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if fine_tuning_history:
        plt.plot(fine_tuning_history.history['loss'], label='Fine-tuning Loss')
        plt.plot(fine_tuning_history.history['val_loss'], label='Fine-tuning Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Initial Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if fine_tuning_history:
        plt.plot(fine_tuning_history.history['accuracy'], label='Fine-tuning Accuracy')
        plt.plot(fine_tuning_history.history['val_accuracy'], label='Fine-tuning Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_history.png')  # Save the chart as a PNG
    plt.show()

# Call the function to plot
plot_training_history(history, fine_tuning_history=history_fine)
