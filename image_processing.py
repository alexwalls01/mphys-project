import numpy as np
import tensorflow as tf
import dm_pix as pix
import jax
from CNN_config import load_CNN_config

CNN_config = load_CNN_config()

RNG = jax.random.PRNGKey(0)

gpu_devices = tf.config.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

def transpose_images(images):
    transposed_images = []
    for image in images:
        if CNN_config['RGB']:
            transpose = np.transpose(np.reshape(image, (3, CNN_config['image_size'][0], CNN_config['image_size'][1])), (1, 2, 0))
        else:
            transpose = np.transpose(np.reshape(image, (1, CNN_config['image_size'][0], CNN_config['image_size'][1])), (1, 2, 0))
        transposed_images.append(transpose)
    transposed_images = np.array(transposed_images)
    return transposed_images

def resize_images(images, target_size):
    resized_images = []
    for image in images:
        resized = tf.image.resize(image, (target_size[0], target_size[1]))
        resized_images.append(resized)
    return np.asarray(resized_images)

def normalise_images(images):
    return (images / 255.0).astype(np.float32)

def preprocess_images(images):
    processed_images = normalise_images(images)
    processed_images = transpose_images(processed_images)
    return processed_images

@jax.jit
def data_augmentation(image):
    print(f"Shape in data_augmentation: {image.shape}, dtype: {image.dtype}")
    augmented_image = image
    if np.random.rand() > 0.5:
        augmented_image = pix.flip_up_down(image=augmented_image)
    if np.random.rand() > 0.5:
        augmented_image = pix.flip_left_right(image=augmented_image)
    num_rotations = np.random.randint(low=0, high=4)
    if num_rotations > 0:
        augmented_image = pix.rot90(k=num_rotations, image=augmented_image)
    return augmented_image

jit_data_augmentation = jax.vmap(data_augmentation, in_axes=0)