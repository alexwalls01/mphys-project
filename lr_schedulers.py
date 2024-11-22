import jax.numpy as jnp

from CNN_config import load_CNN_config, update_CNN_config

update_CNN_config()
CNN_config = load_CNN_config()

def warm_up_cosine_decay(warmup_epochs, total_epochs, initial_learning_rate=CNN_config['learning_rate'], min_learning_rate=CNN_config['min_learning_rate']):
    def scheduler(epoch):
        if epoch < warmup_epochs:
            return initial_learning_rate * (epoch / warmup_epochs)
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        return min_learning_rate + (initial_learning_rate - min_learning_rate) * cosine_decay
    return scheduler