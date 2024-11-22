import jax.numpy as jnp
from flax import linen as nn

from CNN_config import load_CNN_config, update_CNN_config

update_CNN_config()
CNN_config = load_CNN_config()

class LeNet(nn.Module):
    dropout_rate: float = CNN_config['dropout_rate']

    @nn.compact
    def __call__(self, x, train=True):
        
        x = nn.Conv(features=32, kernel_size=(5,5), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(features=32, kernel_size=(3,3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3,3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(features=64, kernel_size=(3,3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(features=CNN_config['num_classes'])(x)
        
        return x

class airbench94(nn.Module):

    @nn.compact
    def __call__(self, x, train=True):
        
        x = nn.Conv(features=24, kernel_size=(3,3))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=CNN_config['num_classes'])(x)
        
        return x