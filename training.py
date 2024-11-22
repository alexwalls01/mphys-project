#!/usr/bin/env python3

from CNN_config import update_CNN_config, load_CNN_config
from CNN import CNN, create_train_state, train_model
from CIFAR10 import load_CIFAR_data, load_CIFAR_annotations
import numpy as np
from datasets import create_JAX_dataset
from architectures import LeNet, airbench94

update_CNN_config()
CNN_config = load_CNN_config()

if CNN_config['architecture'] == 'LeNet':
    CNN = LeNet()
elif CNN_config['architecture'] == 'airbench94':
    CNN = airbench94()

model = CNN
initial_state = create_train_state(model, learning_rate=CNN_config['learning_rate'], weight_decay=CNN_config['weight_decay'])
images, _ = load_CIFAR_data('cifar-10-batches-py/test_batch')
annotations = load_CIFAR_annotations('cifar-10h/data/cifar10h-probs.npy')
train_images, test_images, _ = np.split(images[:-1], [7999, 8999])
train_annotations, test_annotations, _ = np.split(annotations[:-1], [7999, 8999])
train_set = create_JAX_dataset(train_images, train_annotations, batch_size=CNN_config['batch_size'], shuffle=True, augment=True)
test_set = create_JAX_dataset(test_images, test_annotations, batch_size=CNN_config['batch_size'])
trained_state = train_model(initial_state, train_set, test_set, use_wandb=True, use_early_stopping=True, save_model=True)