import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from io import BytesIO
from sklearn.metrics import confusion_matrix
import plotly.express as px
import os
from reportlab.lib.pagesizes import letter # For generating downloadable reports
from reportlab.pdfgen import canvas 
import pdfkit  # For generating downloadable reports
import math
import tempfile
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set page configuration
st.set_page_config(
    page_title="Medical Imaging CNN Trainer",
    page_icon="🧠",  # Use a brain emoji for the favicon
    layout="wide",
)

def normalize_standardize(data, method='Normalization'):
    if method == 'Normalization':
        return data / 255.0
    elif method == 'Standardization':
        mean = np.mean(data, axis=(0, 1, 2, 3), keepdims=True)
        std = np.std(data, axis=(0, 1, 2, 3), keepdims=True)
        return (data - mean) / (std + 1e-7)
    return data

# Function to flatten images
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

# Function to unflatten images
def unflatten_images(flattened_images, original_shape):
    return flattened_images.reshape(flattened_images.shape[0], *original_shape)

# Functions for class imbalance handling
def apply_smote(images, labels, sampling_strategy='auto'):
    smote = SMOTE(sampling_strategy=sampling_strategy)
    flattened_images = flatten_images(images)
    augmented_images, augmented_labels = smote.fit_resample(flattened_images, labels)
    augmented_images = unflatten_images(augmented_images, images.shape[1:])
    return augmented_images, augmented_labels

def apply_random_oversampling(images, labels, sampling_strategy='auto'):
    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    flattened_images = flatten_images(images)
    augmented_images, augmented_labels = ros.fit_resample(flattened_images, labels)
    augmented_images = unflatten_images(augmented_images, images.shape[1:])
    return augmented_images, augmented_labels

def apply_random_undersampling(images, labels, sampling_strategy='auto'):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    flattened_images = flatten_images(images)
    augmented_images, augmented_labels = rus.fit_resample(flattened_images, labels)
    augmented_images = unflatten_images(augmented_images, images.shape[1:])
    return augmented_images, augmented_labels

def apply_adasyn(images, labels, sampling_strategy='auto'):
    adasyn = ADASYN(sampling_strategy=sampling_strategy)
    flattened_images = flatten_images(images)
    augmented_images, augmented_labels = adasyn.fit_resample(flattened_images, labels)
    augmented_images = unflatten_images(augmented_images, images.shape[1:])
    return augmented_images, augmented_labels

# Function to augment images
def augment_images(images, labels, augmentation_params):
    datagen = ImageDataGenerator(
        rotation_range=augmentation_params['rotation_range'],
        width_shift_range=augmentation_params['width_shift_range'],
        height_shift_range=augmentation_params['height_shift_range'],
        shear_range=augmentation_params['shear_range'],
        zoom_range=augmentation_params['zoom_range'],
        horizontal_flip=augmentation_params['horizontal_flip'],
        vertical_flip=augmentation_params['vertical_flip']
    )
    
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(images, labels):
        image = np.expand_dims(image, 0)  # Add batch dimension
        it = datagen.flow(image, batch_size=1)
        for _ in range(augmentation_params['num_augmentations']):
            aug_image = next(it)[0].astype(np.uint8)  # Use next() to get the augmented image
            augmented_images.append(aug_image)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)


# Function to calculate practical limits based on input shape
def calculate_practical_limits(input_shape):
    height, width = input_shape[0], input_shape[1]
    max_conv_layers = 0
    temp_height, temp_width = height, width

    while temp_height > 2 and temp_width > 2:
        max_conv_layers += 1
        temp_height = (temp_height - 2) // 2
        temp_width = (temp_width - 2) // 2

    max_conv_units = min(64, height * width // 16)

    return max_conv_layers, max_conv_units

# Define the function to build and compile the model with user-defined architecture
def build_model(input_shape, num_classes, conv_layers, conv_units, dense_units, dropout_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(conv_units, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    for _ in range(conv_layers - 1):
        model.add(tf.keras.layers.Conv2D(conv_units, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

# Define the function to train the model with user-defined parameters
import io  # Import for capturing model summary

# Define the function to train the model with user-defined parameters
def train_model(train_images, train_labels, val_images, val_labels, input_shape, num_classes, epochs, batch_size, optimizer, learning_rate, conv_layers, conv_units, dense_units, dropout_rate):
    model = build_model(input_shape, num_classes, conv_layers, conv_units, dense_units, dropout_rate)

    # Capture and display the model summary before training
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    summary_string = buffer.getvalue()
    buffer.close()
    
    st.subheader("Model Architecture Summary")
    st.code(summary_string, language='text')  # Display the model summary in Streamlit
    
    # Proceed with optimizer selection
    optimizer_instance = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
        'adadelta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
        'ftrl': tf.keras.optimizers.Ftrl(learning_rate=learning_rate),
        'nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    }.get(optimizer, tf.keras.optimizers.Adam())

    model.compile(optimizer=optimizer_instance,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    class EpochProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            st.session_state.progress_bar.progress((epoch + 1) / epochs)
            st.session_state.epoch_text.write(f"Epoch {epoch + 1}/{epochs}")
            st.session_state.epoch_metrics.metric("Training Accuracy", f"{logs['accuracy']:.4f}")

    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = st.progress(0)
    if 'epoch_text' not in st.session_state:
        st.session_state.epoch_text = st.empty()
    if 'epoch_metrics' not in st.session_state:
        st.session_state.epoch_metrics = st.sidebar.empty()

    with st.spinner("Training in progress..."):
        history = model.fit(train_images, train_labels, epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(val_images, val_labels),
                            callbacks=[EpochProgress()])

    # Save the final training run's performance for leaderboard tracking
    save_training_run(history.history['val_accuracy'][-1], history.history['val_loss'][-1])

    return model, history


# Function to evaluate the model
def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
    st.write(f"Test Loss: {loss:.4f}")
    st.write(f"Test Accuracy: {accuracy:.4f}")
    st.sidebar.metric("Test Accuracy", f"{accuracy:.2f}%")
    st.sidebar.metric("Test Loss", f"{loss:.4f}")


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Function to generate and download PDF report using ReportLab
def generate_report(history):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 100, "Model Performance Report")
    c.drawString(100, height - 120, f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    c.drawString(100, height - 140, f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    c.drawString(100, height - 160, f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    c.drawString(100, height - 180, f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

    c.save()
    pdf_buffer.seek(0)  # Go to the beginning of the buffer

    return pdf_buffer.getvalue()

# Function to save and track model performance for leaderboard
def save_training_run(accuracy, loss):
    st.session_state["runs"] = st.session_state.get("runs", [])
    st.session_state["runs"].append({"accuracy": accuracy, "loss": loss})

# Function to display leaderboard
def show_leaderboard():
    st.subheader("Model Leaderboard")
    if "runs" in st.session_state:
        runs = st.session_state["runs"]
        runs = sorted(runs, key=lambda x: x["accuracy"], reverse=True)
        for i, run in enumerate(runs):
            st.write(f"Run {i + 1}: Accuracy: {run['accuracy']:.4f}, Loss: {run['loss']:.4f}")

# Function to plot training history with Plotly
def plot_training_history(history):
    st.subheader("Training History")
    fig_acc = px.line(
        history.history,
        y=['accuracy', 'val_accuracy'],
        x=range(1, len(history.history['accuracy'])+1),
        labels={'x': 'Epoch', 'y': 'Accuracy'},
        title="Training and Validation Accuracy",
    )
    st.plotly_chart(fig_acc)

    fig_loss = px.line(
        history.history,
        y=['loss', 'val_loss'],
        x=range(1, len(history.history['loss'])+1),
        labels={'x': 'Epoch', 'y': 'Loss'},
        title="Training and Validation Loss",
    )
    st.plotly_chart(fig_loss)

# Function to plot confusion matrix
def plot_confusion_matrix(model, test_images, test_labels):
    st.subheader("Confusion Matrix")
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_labels, pred_labels)
    fig_cm = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(fig_cm)

# Updated Data Augmentation Viewer for 2D and 3D images
def preview_augmentations(image, is_color=True, is_3d=False):
    aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )
    st.write("Augmented Images Preview")

    if is_3d:
        depth_slices = image.shape[0]
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))

        for i in range(5):
            slice_idx = np.random.randint(0, depth_slices)  # Random slice
            slice_image = image[slice_idx]  # Extract 2D slice

            slice_image_exp = np.expand_dims(slice_image, axis=(0, -1))  # Expand dims

            for batch in aug_gen.flow(slice_image_exp, batch_size=1):
                aug_image = batch[0].squeeze()  # Squeeze batch
                ax[i].imshow(aug_image, cmap="gray")
                ax[i].axis('off')
                break  # Augment 1 image per slice

    else:
        image = np.expand_dims(image, axis=0)  # Add batch dimension for augmentation
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))

        for i, batch in enumerate(aug_gen.flow(image, batch_size=1)):
            aug_image = batch[0]
            if aug_image.shape[-1] == 1:  # Grayscale image
                aug_image = np.squeeze(aug_image, axis=-1)
                ax[i].imshow(aug_image, cmap="gray")
            elif aug_image.shape[-1] == 3:  # RGB image
                ax[i].imshow(aug_image.astype('uint8'))
            ax[i].axis('off')
            if i == 4:
                break

    st.pyplot(fig)

import numpy as np
import matplotlib.pyplot as plt
import math
import streamlit as st

def check_class_imbalance(labels, threshold):
    # Flatten the labels array to ensure it's 1D
    labels = np.ravel(labels)
    
    # Ensure labels are non-negative integers
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Labels should be non-negative integers.")
    
    # Count occurrences of each class
    class_counts = np.bincount(labels)
    
    # Calculate imbalance
    total_samples = class_counts.sum()
    imbalance_ratio = class_counts.max() / (class_counts.mean() + 1e-6)
    
    # Check if imbalance is significant
    if imbalance_ratio > threshold:
        return True, class_counts
    else:
        return False, class_counts

def preview_data(train_images, train_labels, val_images, val_labels, test_images, test_labels, dataset, selected_labels, num_samples, imbalance_threshold):
    st.subheader("Preview Data")
    st.write(f"Sample of Images from {dataset} with Selected Labels")

    # Select the correct dataset
    if dataset == "Training":
        images, labels = train_images, train_labels
    elif dataset == "Validation":
        images, labels = val_images, val_labels
    else:
        images, labels = test_images, test_labels

    # Filter images based on the selected labels
    indices = np.concatenate([np.where(labels == label)[0] for label in selected_labels])
    selected_images = images[indices]
    selected_labels_arr = labels[indices]

    # Check for class imbalance
    imbalance_warning, class_counts = check_class_imbalance(labels, imbalance_threshold)
    if imbalance_warning:
        st.markdown(f'<p style="color:red;">Warning: Dataset has significant class imbalance. Class distribution: {dict(enumerate(class_counts))}</p>', unsafe_allow_html=True)
    else:
        st.write(f"Class distribution: {dict(enumerate(class_counts))}")

    # Check if num_samples exceeds available samples
    if num_samples > len(selected_images):
        num_samples = len(selected_images)
        st.warning(f"Number of requested samples exceeds available samples. Showing {num_samples} samples instead.")

    # Randomly select samples
    sampled_indices = np.random.choice(len(selected_images), num_samples, replace=False)
    selected_images = selected_images[sampled_indices]
    selected_labels_arr = selected_labels_arr[sampled_indices]

    # Calculate grid size
    num_rows = int(math.ceil(math.sqrt(num_samples)))
    num_cols = int(math.ceil(num_samples / num_rows))

    # Create a grid layout for displaying images
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    ax = ax.flatten()

    for i in range(num_samples):
        if len(selected_images.shape) == 4:  # For 3D images (e.g., (num_samples, height, width, depth))
            num_slices = selected_images.shape[1]
            slice_idx = np.random.randint(0, num_slices)
            image_slice = selected_images[i, slice_idx, :, :]  # Extract a 2D slice
            ax[i].imshow(image_slice, cmap="gray")
        else:
            ax[i].imshow(selected_images[i].squeeze(), cmap="gray")
        
        # Annotate with the label
        ax[i].set_title(f"Label: {selected_labels_arr[i]}")
        ax[i].axis('off')

    # Hide any unused subplots
    for j in range(num_samples, len(ax)):
        ax[j].axis('off')

    st.pyplot(fig)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_images(images, labels, augmentation_params):
    datagen = ImageDataGenerator(
        rotation_range=augmentation_params['rotation_range'],
        width_shift_range=augmentation_params['width_shift_range'],
        height_shift_range=augmentation_params['height_shift_range'],
        shear_range=augmentation_params['shear_range'],
        zoom_range=augmentation_params['zoom_range'],
        horizontal_flip=augmentation_params['horizontal_flip'],
        vertical_flip=augmentation_params['vertical_flip']
    )

    augmented_images = []
    augmented_labels = []

    # Apply augmentation to the images
    for i in range(len(images)):
        img = images[i].reshape((1,) + images[i].shape)  # reshape for Keras generator
        label = labels[i]

        # Generate augmented images for this sample
        aug_iter = datagen.flow(img, batch_size=1)
        for _ in range(augmentation_params['num_augmentations']):
            aug_img = next(aug_iter)[0]  # Use next() function
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def display_sample_images(images, labels, labels_to_display, num_samples):
    # Ensure that images and labels are numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Initialize subplot variables
    fig, axes = plt.subplots(len(labels_to_display), num_samples, figsize=(15, 3 * len(labels_to_display)))
    
    for label_idx, label in enumerate(labels_to_display):
        # Get indices of images with the current label
        label_indices = np.where(labels == label)[0]
        if len(label_indices) == 0:
            st.write(f"No images found for label {label}.")
            continue

        # Randomly select samples
        sample_indices = np.random.choice(label_indices, size=min(num_samples, len(label_indices)), replace=False)
        sample_images = images[sample_indices]
        
        for i, ax in enumerate(axes[label_idx]):
            if i < len(sample_images):
                img = sample_images[i]
                if img.ndim == 3:  # Handle 3D images if necessary
                    img = img[:, :, img.shape[2] // 2]  # Take the middle slice
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                # Add label text to the subplot
                if i == 0:
                    ax.set_title(f"Label {label}", fontsize=12)
            else:
                ax.axis('off')
                
    st.pyplot(fig)


# Function to download the dataset
def download_data(images, labels, filename, key=None):
    buffer = BytesIO()
    np.savez_compressed(buffer, images=images, labels=labels)
    buffer.seek(0)
    st.download_button(label="Download " + filename, data=buffer, file_name=filename + ".npz", key=key)

# Helper function to convert data to a downloadable file
def create_downloadable_file(data, file_name):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, train_images=data[0], train_labels=data[1])
    buffer.seek(0)
    return buffer.read()

import os
import tempfile
from io import BytesIO
import streamlit as st
import numpy as np
from PIL import Image

# Add this to initialize the session state
#if 'model_bytes' not in st.session_state:
#    st.session_state.model_bytes = None
#if 'report' not in st.session_state:
#    st.session_state.report = None

def main():

    # Initialize session state attributes if they don't exist
    if 'model_bytes' not in st.session_state:
        st.session_state.model_bytes = None
    if 'report_bytes' not in st.session_state:
        st.session_state.report_bytes = None
    if 'training_successful' not in st.session_state:
        st.session_state.training_successful = False
    if 'show_class_imbalance' not in st.session_state:
        st.session_state.show_class_imbalance = False
    if 'balancing_done' not in st.session_state:
        st.session_state['balancing_done'] = False
    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = {}
    if 'processed_dataset' not in st.session_state:
        st.session_state['processed_dataset'] = None

        
    # Display logo image
    image = Image.open('CaptureResized.jpg')
    st.image(image, caption='@2024 All Rights Reserved To Me')

    # App description
    st.markdown("""
#### Welcome to BildSt®
**BildSt®** is a sophisticated, web-based tool built on Streamlit, designed to streamline the exploration, preprocessing, and augmentation of medical imaging datasets. Our platform simplifies complex image analysis tasks, enabling you to focus on what matters most—your data and its potential.

#### Key Features
##### 📁 **Seamless Data Upload and Overview**
Easily upload your medical imaging datasets in `.npz` format. Gain immediate insights into your dataset with detailed information on the shape and size of your training, validation, and test sets.

##### 🔍 **In-Depth Dataset Exploration**
Explore critical aspects of your dataset, including image dimensions and class distribution. Select specific labels and visualize the number of samples to better understand your data's composition.

##### 🔧 **Interactive Data Preprocessing**
Apply preprocessing steps such as normalization and standardization to your images. Customize your preprocessing approach and instantly view the effects on your dataset.

##### 🏷️ **Class Imbalance Handling**
Address class imbalance issues with various techniques like SMOTE, random oversampling, and undersampling. Select the method that best suits your needs and improve the balance of your dataset.

##### 🔄 **Advanced Data Augmentation**
Enhance your dataset through a range of augmentation techniques. Adjust parameters for rotation, shifting, zooming, and flipping to create a diverse set of training images and boost model performance.

#### 📊 **Model Training and Evaluation**
Train convolutional neural networks (CNNs) with configurable parameters such as the number of layers, units, dropout rate, and more. Monitor training progress, evaluate model performance, and visualize results through interactive plots and metrics.

##### 📥 **Model and Report Download**
Download your trained models and detailed performance reports. Save your work in convenient formats for further use or integration into other systems.

##### ⚙️ **User-Friendly Interface**
Navigate through a streamlined interface with interactive sliders, checkboxes, and buttons. Customize your settings and visualize updates in real-time, making data analysis straightforward and intuitive.

Experience the power of advanced data preprocessing and model training with BildSt®. Simplify your workflow and achieve meaningful insights directly from your browser.
""")


    file = st.file_uploader("Upload a .npz dataset file", type=["npz"])
    if file is not None:
        with np.load(file) as data:
            train_images = data['train_images']
            train_labels = data['train_labels']
            val_images = data['val_images']
            val_labels = data['val_labels']
            test_images = data['test_images']
            test_labels = data['test_labels']

        st.write(f"Train set: {train_images.shape}, Train labels: {train_labels.shape}")
        st.write(f"Validation set: {val_images.shape}, Validation labels: {val_labels.shape}")
        st.write(f"Test set: {test_images.shape}, Test labels: {test_labels.shape}")

        # Store the dataset in session state after the file is uploaded
        if st.session_state['processed_dataset'] is None:
            st.session_state['processed_dataset'] = (train_images, train_labels)

        train_images = train_images.astype('float32') / 255.0
        val_images = val_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0

        # Use the dataset from session state for further processing
        processed_images, processed_labels = st.session_state['processed_dataset']

        input_shape = train_images.shape[1:]
        num_classes = len(np.unique(train_labels))

        max_conv_layers, max_conv_units = calculate_practical_limits(input_shape)

        st.sidebar.header("Preview Options")
        dataset = st.sidebar.selectbox("Select Dataset", ["Training", "Validation", "Test"])

        # Get unique labels for selected dataset
        if dataset == "Training":
            unique_labels = np.unique(train_labels)
            label_samples = [len(np.where(train_labels == label)[0]) for label in unique_labels]
        elif dataset == "Validation":
            unique_labels = np.unique(val_labels)
            label_samples = [len(np.where(val_labels == label)[0]) for label in unique_labels]
        else:
            unique_labels = np.unique(test_labels)
            label_samples = [len(np.where(test_labels == label)[0]) for label in unique_labels]

        selected_labels = []
        for label in unique_labels:
            if st.sidebar.checkbox(f"Label {label}", value=False):
                selected_labels.append(label)

        if selected_labels:
            max_samples_list = [label_samples[np.where(unique_labels == label)[0][0]] for label in selected_labels]
            max_samples = max(max_samples_list)

            if dataset == "Training":
                total_samples = len(np.concatenate([np.where(train_labels == label)[0] for label in selected_labels]))
            elif dataset == "Validation":
                total_samples = len(np.concatenate([np.where(val_labels == label)[0] for label in selected_labels]))
            else:
                total_samples = len(np.concatenate([np.where(test_labels == label)[0] for label in selected_labels]))

            num_samples = st.sidebar.slider("Number of Samples to Display", min_value=1, max_value=max_samples, value=min(16, total_samples))
            imbalance_threshold = st.sidebar.slider("Class Imbalance Ratio Threshold", min_value=1.0, max_value=10.0, value=2.0, step=0.1)

            st.sidebar.write(f"Total samples for selected labels: {total_samples}")

            if st.sidebar.button("Preview Data"):
                preview_data(train_images, train_labels, val_images, val_labels, test_images, test_labels, dataset, selected_labels, num_samples, imbalance_threshold)
                # Check class imbalance for the selected dataset and update session state
                if dataset == "Training":
                    st.session_state.show_class_imbalance = check_class_imbalance(train_labels, imbalance_threshold)
                elif dataset == "Validation":
                    st.session_state.show_class_imbalance = check_class_imbalance(val_labels, imbalance_threshold)
                else:
                    st.session_state.show_class_imbalance = check_class_imbalance(test_labels, imbalance_threshold)
        else:
            st.sidebar.write("No labels selected.")

        st.sidebar.header("Data Preprocessing")
        preprocessing_method = st.sidebar.selectbox(
            "Normalize/Standardize",
            options=['None', 'Normalization', 'Standardization']
        )
        if preprocessing_method != 'None':
            if st.sidebar.button("Apply"):
                with st.spinner('Applying preprocessing...'):
                    # Initialize the progress bar
                    progress_bar = st.sidebar.progress(0)
                    # Perform preprocessing
                    train_images = normalize_standardize(train_images, preprocessing_method)
                    val_images = normalize_standardize(val_images, preprocessing_method)
                    test_images = normalize_standardize(test_images, preprocessing_method)
                    # Complete the progress bar
                    progress_bar.progress(100)
                    st.sidebar.success(f"{preprocessing_method} applied successfully.")

                    # Update the processed dataset
                    #st.session_state['processed_dataset'] = (processed_images, processed_labels)
                    st.session_state['processed_dataset'] = (train_images, train_labels)

                    # Provide download link for processed data
                    processed_data_bytes = create_downloadable_file((train_images, train_labels), "processed_data.npz")
                    st.download_button(
                        label="Download Processed Data",
                        data=processed_data_bytes,
                        file_name="processed_data.npz",
                        mime="application/octet-stream"
                    )

        if st.session_state.show_class_imbalance:
            imbalance_method = st.sidebar.selectbox(
                "Select Class Imbalance Handling Technique",
                options=['None', 'SMOTE', 'Random Oversampling', 'Random Undersampling', 'ADASYN']
            )
            if imbalance_method != 'None':
                sampling_strategy_options = ['auto', 'minority', 'majority', 'not majority', 'all']
                sampling_strategy = st.sidebar.selectbox("Sampling Strategy", sampling_strategy_options, index=sampling_strategy_options.index('auto'))
                
                if st.sidebar.button("Apply Class Imbalance Handling"):
                    with st.spinner("Applying balancing technique..."):
                        # Initialize a progress bar
                        progress_bar = st.progress(0)
                        total_steps = 100
                        for i in range(total_steps):
                            # Simulate processing time
                            time.sleep(0.05)
                            progress_bar.progress(i + 1)
                        
                        # Start the actual balancing process
                        processed_images, processed_labels = st.session_state['processed_dataset']
                        if imbalance_method == 'SMOTE':
                            train_images, train_labels = apply_smote(processed_images, processed_labels, sampling_strategy=sampling_strategy)
                        elif imbalance_method == 'Random Oversampling':
                            train_images, train_labels = apply_random_oversampling(processed_images, processed_labels, sampling_strategy=sampling_strategy)
                        elif imbalance_method == 'Random Undersampling':
                            train_images, train_labels = apply_random_undersampling(processed_images, processed_labels, sampling_strategy=sampling_strategy)
                        elif imbalance_method == 'ADASYN':
                            train_images, train_labels = apply_adasyn(processed_images, processed_labels, sampling_strategy=sampling_strategy)
                        
                        # Update session state with the balanced dataset
                        #st.session_state['processed_dataset'] = (processed_images, processed_labels)
                        # Update session state with the balanced dataset
                        st.session_state['processed_dataset'] = (train_images, train_labels)

                        # Provide download link for balanced data
                        balanced_data_bytes = create_downloadable_file((train_images, train_labels), "balanced_data.npz")
                        st.download_button(
                            label="Download Balanced Data",
                            data=balanced_data_bytes,
                            file_name="balanced_data.npz",
                            mime="application/octet-stream"
                        )

                        # Remove the progress bar and show success message
                        st.success(f"Class imbalance handling with {imbalance_method} applied successfully.")

                        # Update the dataset with the new balanced version
                        #balanced_dataset_name = f'{dataset}_Balanced'
                        #st.session_state.datasets[balanced_dataset_name] = (train_images, train_labels)
                        #st.session_state['balancing_done'] = True
                        #download_data(train_images, train_labels, balanced_dataset_name, key=balanced_dataset_name)
        
        st.sidebar.subheader("Augmentation Parameters")
        augmentation_params = {
            'rotation_range': st.sidebar.slider("Rotation Range", 0, 360, 0),
            'width_shift_range': st.sidebar.slider("Width Shift Range", 0.0, 0.5, 0.0),
            'height_shift_range': st.sidebar.slider("Height Shift Range", 0.0, 0.5, 0.0),
            'shear_range': st.sidebar.slider("Shear Range", 0.0, 0.5, 0.0),
            'zoom_range': st.sidebar.slider("Zoom Range", 0.0, 0.5, 0.0),
            'horizontal_flip': st.sidebar.checkbox("Horizontal Flip"),
            'vertical_flip': st.sidebar.checkbox("Vertical Flip"),
            'num_augmentations': st.sidebar.slider("Number of Augmentations per Image", 1, 10, 1)
        }
        if st.sidebar.button("Apply Augmentation"):
            with st.spinner("Starting data augmentation..."):
                processed_images, processed_labels = st.session_state['processed_dataset']
                # Retrieve the dataset to augment from session state
                processed_images, processed_labels = st.session_state['processed_dataset']
                # Apply augmentation on the processed images and labels
                augmented_images, augmented_labels = augment_images(processed_images, processed_labels, augmentation_params)

            st.success("Data augmentation complete!")

            # Update the processed dataset with augmented data
            st.session_state['processed_dataset'] = (augmented_images, augmented_labels)

            # Provide download link for augmented data
            augmented_data_bytes = create_downloadable_file((augmented_images, augmented_labels), "augmented_data.npz")
            st.download_button(
                label="Download Augmented Data",
                data=augmented_data_bytes,
                file_name="augmented_data.npz",
                mime="application/octet-stream"
            )

            # Ensure augmented data exists before trying to display it
            if 'processed_dataset' in st.session_state:
                augmented_images, augmented_labels = st.session_state['processed_dataset']

                # Automatically display 5 random samples for each label (0 and 1)
                st.subheader("Sample Augmented Images")
                display_sample_images(augmented_images, augmented_labels, [0, 1], num_samples=5)

        st.sidebar.header("Model Configuration")
        conv_layers = st.sidebar.slider("Number of Convolutional Layers", min_value=1, max_value=max_conv_layers, value=3)
        conv_units = st.sidebar.slider("Number of Convolutional Units", min_value=16, max_value=max_conv_units, value=32)
        dense_units = st.sidebar.slider("Number of Dense Units", min_value=16, max_value=128, value=64)
        dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.0)
        batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32)
        optimizer = st.sidebar.selectbox("Optimizer", options=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'ftrl', 'nadam'])
        learning_rate = st.sidebar.slider("Learning Rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f")
        epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=500, value=10)

        #is_3d = st.sidebar.checkbox("3D Data", value=False)
        #is_color = st.sidebar.checkbox("Color Images", value=False)
        #st.sidebar.button("Preview Augmentations", on_click=preview_augmentations, args=(train_images[0], is_color, is_3d))

        training_successful = False

        # Add a select box for choosing data type (processed or original)
        data_choice = st.sidebar.selectbox("Use Data Type", ["Original", "Processed"])

        if st.button("Train CNN"):
            st.write("Training model...")

            # Choose dataset based on user selection
            if data_choice == "Processed":
                if 'processed_dataset' in st.session_state:
                    processed_images, processed_labels = st.session_state['processed_dataset']
                    train_images, train_labels = processed_images, processed_labels
                else:
                    st.error("No processed data available. Please preprocess and augment your data first.")
                    st.session_state.training_successful = False
                    return
            else:
                # Use original dataset
                train_images = train_images.astype('float32') / 255.0
                train_labels = train_labels
                val_images = val_images.astype('float32') / 255.0
                test_images = test_images.astype('float32') / 255.0

            try:
                # Train the model
                model, history = train_model(train_images, train_labels, val_images, val_labels, input_shape, num_classes, epochs, batch_size, optimizer, learning_rate, conv_layers, conv_units, dense_units, dropout_rate)
                st.write("Training complete.")

                # Plot training history and evaluate model
                plot_training_history(history)
                evaluate_model(model, test_images, test_labels)
                plot_confusion_matrix(model, test_images, test_labels)

                # Save and store the model and report
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_model_file:
                    model.save(tmp_model_file.name)
                    tmp_model_path = tmp_model_file.name

                try:
                    with open(tmp_model_path, "rb") as f:
                        st.session_state.model_bytes = f.read()  # Save model to session state
                except Exception as e:
                    st.error(f"Error reading model: {e}")

                try:
                    st.session_state.report_bytes = generate_report(history)  # Save report to session state
                except Exception as e:
                    st.error(f"Error generating report: {e}")

                os.remove(tmp_model_path)  # Clean up temporary file

                st.session_state.training_successful = True

                # Show download buttons
                if st.session_state.model_bytes:
                    st.download_button(
                        label="Download Trained Model",
                        data=st.session_state.model_bytes,
                        file_name="cnn_model.h5",
                        mime="application/octet-stream"
                    )

                if st.session_state.report_bytes:
                    st.download_button(
                        label="Download PDF Report",
                        data=st.session_state.report_bytes,
                        file_name="model_report.pdf",
                        mime="application/pdf"
                    )

                show_leaderboard()

            except Exception as e:
                st.error(f"Model training failed. Please check your inputs. Error: {e}")
                st.session_state.training_successful = False

            if st.session_state.training_successful:
                st.success("Model trained successfully!")
            else:
                st.error("Model training failed. Please check your inputs.")

if __name__ == "__main__":
    main()
