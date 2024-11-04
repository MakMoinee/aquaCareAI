import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import xml.etree.ElementTree as ET

# Define paths to your dataset
image_dir = './train/images'
label_dir = './train/labels'
img_size = (299, 299)  # Xception expects 299x299 images
batch_size = 32

# Function to parse labels from XML (adjust if labels are in a different format)
def parse_label_from_xml(label_path):
    tree = ET.parse(label_path)
    root = tree.getroot()
    # Example assumes binary classification with 'disease' and 'healthy' labels
    for obj in root.findall('object'):
        label = obj.find('name').text
        return 1 if label == 'disease' else 0
    return 0

# Function to create dataset
def create_dataset(image_dir, label_dir, subset):
    image_paths = sorted([os.path.join(image_dir, subset, fname) for fname in os.listdir(os.path.join(image_dir, subset)) if fname.endswith('.jpg')])
    label_paths = sorted([os.path.join(label_dir, subset, fname.replace('.jpg', '.xml')) for fname in os.listdir(os.path.join(image_dir, subset)) if fname.endswith('.jpg')])

    images = []
    labels = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(img_path)), img_size)
        label = parse_label_from_xml(lbl_path)
        images.append(img)
        labels.append(label)

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
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Unfreeze the base model and fine-tune
for layer in base_model.layers:
    layer.trainable = True

# Re-compile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

# Save the model in the SavedModel format
model_save_path = 'result.h5'
model.save(model_save_path, save_format='h5')

print(f"Model saved to {model_save_path}")
