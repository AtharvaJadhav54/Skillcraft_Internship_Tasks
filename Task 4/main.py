import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import kagglehub
from pathlib import Path
import shutilwh

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("✅ Libraries imported successfully.")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


# ----------------------------------------------------------------------------
# 2. DOWNLOAD AND PREPARE THE DATASET
# ----------------------------------------------------------------------------
print("\n🚀 Step 2: Downloading and preparing the dataset...")

# Download the dataset using KaggleHub
dataset_path_root = Path(kagglehub.dataset_download("gti-upm/leapgestrecog"))

# --- Dynamically locate the correct data directory ---
original_data_dir = None
potential_dirs = list(dataset_path_root.rglob('00'))
if potential_dirs:
    original_data_dir = potential_dirs[0].parent
    print(f"✅ Automatically located data directory: {original_data_dir}")
else:
    print(f"❌ Error: Could not automatically locate the subject folders (e.g., '00') within {dataset_path_root}.")
    original_data_dir = dataset_path_root / 'leapgestrecog'
    print(f"Falling back to default path: {original_data_dir}")


# The dataset is structured as subject/gesture/image. We need to restructure
# it to gesture/image for tf.keras.utils.image_dataset_from_directory.
processed_data_dir = Path('./leapgestrecog_processed')

if processed_data_dir.exists():
    print(f"Found existing processed directory. Deleting it to ensure a clean slate.")
    shutil.rmtree(processed_data_dir)
processed_data_dir.mkdir()

print(f"Restructuring data from {original_data_dir} to {processed_data_dir}...")

# --- OPTIMIZATION: Use a minimal subset of the data for maximum speed ---
# We will use data from only 1 subject.
NUM_SUBJECTS_TO_USE = 1

if not original_data_dir.exists():
    print(f"❌ Error: The source directory {original_data_dir} was not found after download.")
    print("Please check the KaggleHub download path and dataset structure.")
else:
    subject_dirs = sorted([d for d in original_data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} subjects. Using the first {NUM_SUBJECTS_TO_USE}.")
    
    # Iterate through a subset of subject directories
    for subject_dir in subject_dirs[:NUM_SUBJECTS_TO_USE]:
        print(f"  Processing subject: {subject_dir.name}")
        for gesture_dir in subject_dir.iterdir():
            if not gesture_dir.is_dir():
                continue
            
            class_name_parts = gesture_dir.name.split('_', 1)
            clean_class_name = class_name_parts[1] if len(class_name_parts) > 1 else class_name_parts[0]

            dest_dir = processed_data_dir / clean_class_name
            dest_dir.mkdir(exist_ok=True)
            
            for img_path in gesture_dir.glob('*.png'):
                shutil.copy(img_path, dest_dir)
    print("✅ Data restructuring complete.")

# Define image parameters and use the new processed directory
data_dir = processed_data_dir
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32

# Create the training dataset using Keras utility from the restructured data
print("\nLoading training data from processed directory...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Create the validation dataset
print("Loading validation data from processed directory...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get class names from the dataset
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n✅ Found {num_classes} gesture classes:")
# Sort class names for consistent ordering
class_names.sort()
print(class_names)


# ----------------------------------------------------------------------------
# 3. VISUALIZE THE DATA
# ----------------------------------------------------------------------------
print("\n🚀 Step 3: Visualizing the data...")

plt.figure(figsize=(12, 8))
plt.suptitle("Sample Images from the Dataset", fontsize=16)
for images, labels in train_ds.take(1):
    for i in range(min(9, BATCH_SIZE)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ----------------------------------------------------------------------------
# 4. CONFIGURE DATASET FOR PERFORMANCE
# ----------------------------------------------------------------------------
print("\n🚀 Step 4: Configuring dataset for performance...")

# Use buffered prefetching to load images from disk without I/O blocking
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("✅ Datasets configured with caching and prefetching.")


# ----------------------------------------------------------------------------
# 5. BUILD THE MODEL USING TRANSFER LEARNING (MobileNetV2)
# ----------------------------------------------------------------------------
print("\n🚀 Step 5: Building the model with MobileNetV2...")

# Define input shape
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create the base model from the pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

print(f"Number of layers in the base model: {len(base_model.layers)}")

# Create a data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Create the full model by adding a classification head
model = Sequential([
    layers.Input(shape=IMG_SHAPE),
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()
print("✅ Model built and compiled successfully.")


# ----------------------------------------------------------------------------
# 6. TRAIN THE MODEL
# ----------------------------------------------------------------------------
print("\n🚀 Step 6: Starting initial model training (feature extraction)...")

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_gesture_model.keras', save_best_only=True)

# --- OPTIMIZATION: Reduce epochs for maximum speed ---
initial_epochs = 3
history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds,
                    callbacks=[early_stopping, model_checkpoint])


# ----------------------------------------------------------------------------
# 7. VISUALIZE INITIAL TRAINING RESULTS
# ----------------------------------------------------------------------------
print("\n🚀 Step 7: Visualizing initial training results...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# ----------------------------------------------------------------------------
# 8. FINE-TUNING THE MODEL
# ----------------------------------------------------------------------------
print("\n🚀 Step 8: Fine-tuning the model...")

base_model.trainable = True
# --- OPTIMIZATION: Unfreeze fewer layers for faster fine-tuning ---
fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['accuracy'])

model.summary()
print(f"✅ Model re-compiled for fine-tuning. {len(model.trainable_variables)} trainable variables.")

# --- OPTIMIZATION: Reduce epochs for maximum speed ---
fine_tune_epochs = 2
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds,
                         callbacks=[early_stopping, model_checkpoint])

# ----------------------------------------------------------------------------
# 9. EVALUATE THE FINAL MODEL
# ----------------------------------------------------------------------------
print("\n🚀 Step 9: Evaluating the final model...")

final_model = tf.keras.models.load_model('best_gesture_model.keras')
loss, accuracy = final_model.evaluate(val_ds)
print(f'✅ Final Model Validation Accuracy: {accuracy*100:.2f}%')

print("Generating predictions for confusion matrix...")
y_pred = []
y_true = []
for image_batch, label_batch in val_ds:
   y_true.extend(label_batch)
   preds = final_model.predict(image_batch, verbose=0)
   y_pred.extend(np.argmax(preds, axis=-1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# ----------------------------------------------------------------------------
# 10. PREDICT ON A NEW IMAGE
# ----------------------------------------------------------------------------
print("\n🚀 Step 10: Function to predict on a new image...")

def predict_new_image(model, img_path, class_names):
    """Loads an image, preprocesses it, and predicts the gesture."""
    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence

print("\nTesting prediction function on a sample image from the dataset...")
try:
    sample_image_path = list(data_dir.glob('*/*.png'))[0]
    predict_new_image(final_model, str(sample_image_path), class_names)
    print("\nTo test with your own image in Colab, upload a file and pass its path to `predict_new_image`.")
except (IndexError, FileNotFoundError):
    print("Could not find a sample image to test.")


print("\n🎉🎉🎉 SCRIPT COMPLETED SUCCESSFULLY! 🎉🎉🎉")
