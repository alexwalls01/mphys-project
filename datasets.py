import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from CNN_config import load_CNN_config
from image_processing import preprocess_images, jit_data_augmentation

CNN_config = load_CNN_config()

AUTOTUNE = tf.data.AUTOTUNE

gpu_devices = tf.config.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

def create_JAX_dataset(images, labels, batch_size=CNN_config['batch_size'], shuffle=False, augment=False):
    images = preprocess_images(images)
    remainder = np.remainder(len(images), batch_size)
    if remainder != 0:
        images = images[:-remainder]
        labels = labels[:-remainder]
        print(f'Dropped {remainder} images from dataset')
    # Take images and labels as NumPy arrays and create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        # Shuffle dataset with buffer size equal to number of elements
        dataset = dataset.shuffle(len(images))
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    # Apply dynamic augmentation if training data
    if augment:
        def augment_fn(images, label):
            # Convert TensorFlow Tensors to NumPy arrays, apply JAX augmentations, and convert back
            augmented_image = tf.numpy_function(
                func=lambda img: jit_data_augmentation(img).astype(np.float32),
                inp=[images],
                Tout=tf.float32
            )
            return augmented_image, label
        dataset = dataset.map(augment_fn, num_parallel_calls=AUTOTUNE)
    # Prefetch the dataset
    dataset = dataset.prefetch(AUTOTUNE)
    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(dataset)

def create_monte_carlo_set(images, true_posterior_probs, m):
    calibration_images = []
    calibration_labels = []
    for i in range(0, len(images)):
         for j in range(0, m):
            label = np.zeros(len(true_posterior_probs[i]))
            label[np.where(true_posterior_probs[i] == np.random.choice(true_posterior_probs[i], p=true_posterior_probs[i]))[0][0]] = 1
            calibration_images.append(images[i])
            calibration_labels.append(label)
    return np.asarray(calibration_images), np.asarray(calibration_labels)

def get_images_and_labels(dataset):
    images, labels = tuple(zip(*dataset))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels