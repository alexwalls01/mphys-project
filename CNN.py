import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import numpy as np
import wandb
import pickle

from CNN_config import load_CNN_config, update_CNN_config
from architectures import LeNet, airbench94
from lr_schedulers import warm_up_cosine_decay
from datasets import get_images_and_labels, create_JAX_dataset, create_monte_carlo_set

update_CNN_config()
CNN_config = load_CNN_config()

if CNN_config['architecture'] == 'LeNet':
    CNN = LeNet()
elif CNN_config['architecture'] == 'airbench94':
    CNN = airbench94()

if CNN_config['learning_rate_scheduler'] == 'warmup_cosine':
    SCHEDULER = warm_up_cosine_decay(CNN_config['warmup_epochs'], CNN_config['num_epochs'])

RNG = jax.random.PRNGKey(0)

def init_adamw(params, learning_rate=CNN_config['learning_rate'], beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=CNN_config['weight_decay']):
    m = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    v = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    return {
        'params': params,
        'm': m,
        'v': v,
        'learning_rate': learning_rate,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'weight_decay': weight_decay,
        'step': 0
    }

def update_adamw(opt_state, grads):
    step = opt_state['step'] + 1
    lr_t = opt_state['learning_rate'] * jnp.sqrt(1 - opt_state['beta2']**step) / (1 - opt_state['beta1']**step)
    
    # Compute first and second moment estimates
    m = jax.tree_map(lambda m, g: opt_state['beta1'] * m + (1 - opt_state['beta1']) * g, opt_state['m'], grads)
    v = jax.tree_map(lambda v, g: opt_state['beta2'] * v + (1 - opt_state['beta2']) * (g**2), opt_state['v'], grads)
    
    updates = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + opt_state['eps']), m, v)

    # Update parameters with decoupled weight decay
    def apply_update(param_name, param, update):
        # No decay for biases or BatchNorm
        if "bias" in param_name or "batch_stats" in param_name:
            return param - lr_t * update
        return param - lr_t * update - lr_t * opt_state['weight_decay'] * param
    
    new_params = unfreeze(opt_state['params'])
    updates_dict = unfreeze(updates)

    # Apply updates
    for key, param in new_params.items():
        new_params[key] = jax.tree_map(
            lambda p, u, k=key: apply_update(k, p, u),
            param,
            updates_dict[key]
        )

    return {
        'params': freeze(new_params),
        'm': m,
        'v': v,
        'learning_rate': opt_state['learning_rate'],
        'beta1': opt_state['beta1'],
        'beta2': opt_state['beta2'],
        'eps': opt_state['eps'],
        'weight_decay': opt_state['weight_decay'],
        'step': step
    }

def create_train_state(model, learning_rate=CNN_config['learning_rate'], weight_decay=CNN_config['weight_decay']):
    dropout_rng, init_rng = jax.random.split(RNG)
    variables = model.init({'params': init_rng, 'dropout': dropout_rng}, jnp.ones(CNN_config['input_shape']))
    params = variables['params']
    batch_stats = variables['batch_stats']
    opt_state = init_adamw(params, learning_rate=learning_rate, weight_decay=weight_decay)
    return {
        'params': params,
        'batch_stats': batch_stats,
        'opt_state': opt_state,
        'learning_rate': learning_rate,
    }

def update_learning_rate(state, new_learning_rate):
    updated_state = state.copy()
    updated_state['learning_rate'] = new_learning_rate

    opt_state = state['opt_state']
    updated_opt_state = jax.tree_map(
        lambda x: x._replace(learning_rate=new_learning_rate) if hasattr(x, 'learning_rate') else x,
        opt_state
    )
    updated_state['opt_state'] = updated_opt_state
    return updated_state

def compute_loss(logits, labels):
    log_probs = jax.nn.log_softmax(logits)
    # Cross-entropy loss for soft labels
    loss = -jnp.mean(jnp.sum(labels * log_probs, axis=-1))
    return loss

def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == jnp.argmax(labels, axis=-1))

@jax.jit
def train_step(state, batch):

    images, labels = batch

    params = state['params']
    batch_stats = state['batch_stats']

    def loss_fn(params):
        model = CNN
        logits, new_model_state = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            images,
            mutable=['batch_stats'],
            rngs={'dropout': jax.random.PRNGKey(0)}
        )
        loss = compute_loss(logits, labels)
        accuracy = compute_accuracy(logits, labels)
        return loss, {'accuracy': accuracy, 'batch_stats': new_model_state['batch_stats']}

    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    state['opt_state'] = update_adamw(state['opt_state'], grads)
    state['params'] = state['opt_state']['params']
    state['batch_stats'] = aux['batch_stats']

    accuracy = aux['accuracy']

    return state, loss, accuracy

def eval_step(state, batch):
    images, labels = batch
    params = state['params']
    batch_stats = state['batch_stats']
    model = CNN
    logits = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        images,
        mutable=False,
        train=False 
    )
    loss = compute_loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)
    return loss, accuracy

def train_model(state, train_set, test_set, num_epochs=CNN_config['num_epochs'], use_wandb=True, use_early_stopping=True, save_model=False):
    run_id = wandb.util.generate_id()
    run_directory = CNN_config['run_directory']

    if use_wandb:
        wandb.init(
            project=CNN_config['wandb_project'],
            id=run_id,
            resume='never',
            dir=run_directory,

            config={
            'architecture': CNN_config['architecture'],
            'learning_rate_scheduler': CNN_config['learning_rate_scheduler'],
            'warmup_epochs': CNN_config['warmup_epochs'],
            'learning_rate': CNN_config['learning_rate'],
            'min_learning_rate': CNN_config['min_learning_rate'],
            'weight_decay': CNN_config['weight_decay'],
            'dropout_rate': CNN_config['dropout_rate'],
            'num_epochs': CNN_config['num_epochs'],
            'batch_size': CNN_config['batch_size'],
            'architecture': 'CNN',
            'optimizer': 'adamw',
            }
        )
    
    if use_early_stopping:
        early_stopping = {
            'patience': CNN_config['patience'],
            'best_metric': float('inf'),
            'epochs_without_improvement': 0,
            'best_params': None,
        }

    train_acc, train_loss, val_acc, val_loss = [], [], [], []

    for epoch in range(0, num_epochs):
        train_batch_loss, train_batch_accuracy = [], []
        val_batch_loss, val_batch_accuracy = [], []

        if CNN_config['learning_rate_scheduler'] != False:
                learning_rate = SCHEDULER(epoch)
                state = update_learning_rate(state, learning_rate)

        for batch in train_set:
            state, loss, acc = train_step(state, batch)
            train_batch_loss.append(loss)
            train_batch_accuracy.append(acc)

        for batch in test_set:
            loss, acc = eval_step(state, batch)
            val_batch_loss.append(loss)
            val_batch_accuracy.append(acc)

        # Loss for the current epoch
        epoch_train_loss = np.mean(train_batch_loss)
        epoch_val_loss = np.mean(val_batch_loss)

        # Accuracy for the current epoch
        epoch_train_acc = np.mean(train_batch_accuracy)
        epoch_val_acc = np.mean(val_batch_accuracy)

        if use_early_stopping:
            if epoch_val_loss < early_stopping['best_metric']:
                early_stopping['best_metric'] = epoch_val_loss
                early_stopping['best_params'] = state['params']
                early_stopping['epochs_without_improvement'] = 0
            else:
                early_stopping['epochs_without_improvement'] += 1
            if early_stopping['epochs_without_improvement'] >= early_stopping['patience']:
                state['params'] = early_stopping['best_params']
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if use_wandb:
            wandb.log({'train acc': epoch_train_acc, 'train loss': epoch_train_loss, 'test acc': epoch_val_acc, 'test loss': epoch_val_loss})
        
        train_loss.append([epoch_train_loss])
        train_acc.append([epoch_train_acc])
        val_loss.append([epoch_val_loss])
        val_acc.append([epoch_val_acc])
        
        print(
            f'Epoch {epoch}, train loss: {epoch_train_loss:.5f}, train acc: {epoch_train_acc:.5f}, test loss: {epoch_val_loss:.5f}, test acc: {epoch_val_acc:.5f} '
        )
    
    if use_wandb:
        wandb.finish()
    
    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    val_loss = np.array(val_loss)
    val_acc = np.array(val_acc)
    run_data = np.hstack((train_loss, train_acc, val_loss, val_acc))

    run_data_path = run_directory + '/run_' + run_id + '.csv'
    np.savetxt(run_data_path, run_data)

    if save_model:
        save_model_params(state, run_directory + '/run_' + run_id + '.pkl')

    return state

# Made a separate function for now - could have used recursion but worried about memory usage
def RT4U(state, train_set, test_set, calibration_set, m=CNN_config['m'], num_epochs=CNN_config['num_epochs'], use_wandb=True, use_early_stopping=True, save_model=False):
    run_id = wandb.util.generate_id()
    run_directory = CNN_config['run_directory']

    if use_wandb:
        wandb.init(
            project=CNN_config['wandb_project'],
            id=run_id,
            resume='never',
            dir=run_directory,

            config={
            'architecture': CNN_config['architecture'],
            'learning_rate_scheduler': CNN_config['learning_rate_scheduler'],
            'warmup_epochs': CNN_config['warmup_epochs'],
            'learning_rate': CNN_config['learning_rate'],
            'min_learning_rate': CNN_config['min_learning_rate'],
            'weight_decay': CNN_config['weight_decay'],
            'dropout_rate': CNN_config['dropout_rate'],
            'num_epochs': CNN_config['num_epochs'],
            'batch_size': CNN_config['batch_size'],
            'architecture': 'CNN',
            'optimizer': 'adamw',
            }
        )
    
    if use_early_stopping:
        early_stopping = {
            'patience': 10,
            'best_metric': float('inf'),
            'epochs_without_improvement': 0,
            'best_params': None,
        }
        
    if CNN_config['learning_rate_scheduler'] == 'warmup_cosine':
        scheduler = warm_up_cosine_decay(CNN_config['warmup_epochs'], CNN_config['num_epochs'])
    
    train_images, _ = get_images_and_labels(train_set)
    calibration_images, _ = get_images_and_labels(calibration_set)

    train_softmax, calibration_softmax = [], []

    for epoch in range(0, num_epochs):
        train_batch_loss, train_batch_accuracy = [], []
        val_batch_loss, val_batch_accuracy = [], []

        if CNN_config['learning_rate_scheduler'] != False:
                learning_rate = scheduler(epoch)
                state = update_learning_rate(state, learning_rate)

        for batch in train_set:
            state, loss, acc = train_step(state, batch)
            train_batch_loss.append(loss)
            train_batch_accuracy.append(acc)

        for batch in test_set:
            loss, acc = eval_step(state, batch)
            val_batch_loss.append(loss)
            val_batch_accuracy.append(acc)

        # Loss for the current epoch
        epoch_train_loss = np.mean(train_batch_loss)
        epoch_val_loss = np.mean(val_batch_loss)

        # Accuracy for the current epoch
        epoch_train_acc = np.mean(train_batch_accuracy)
        epoch_val_acc = np.mean(val_batch_accuracy)

        # Softmax for the current epoch
        epoch_train_softmax = predict(state['params'], train_images)
        epoch_calibration_softmax = predict(state['params'], calibration_images)
        
        train_softmax.append(epoch_train_softmax)
        calibration_softmax.append(epoch_calibration_softmax)

        if use_early_stopping:
            if epoch_val_loss < early_stopping['best_metric']:
                early_stopping['best_metric'] = epoch_val_loss
                early_stopping['best_params'] = state['params']
                early_stopping['epochs_without_improvement'] = 0
            else:
                early_stopping['epochs_without_improvement'] += 1
            if early_stopping['epochs_without_improvement'] >= early_stopping['patience']:
                state['params'] = early_stopping['best_params']
                print(f'Early stopping at epoch {epoch}')
                break
        
        if use_wandb:
            wandb.log({'train acc': epoch_train_acc, 'train loss': epoch_train_loss, 'test acc': epoch_val_acc, 'test loss': epoch_val_loss})
        
        print(
            f'Epoch {epoch}, train loss: {epoch_train_loss:.5f}, train acc: {epoch_train_acc:.5f}, test loss: {epoch_val_loss:.5f}, test acc: {epoch_val_acc:.5f} '
        )
    
    if use_wandb:
        wandb.finish()

    new_train_labels = np.mean(np.array(train_softmax), axis=0)
    new_calibration_labels = np.mean(np.array(calibration_softmax), axis=0)

    new_train_set = create_JAX_dataset(train_images, new_train_labels, shuffle=True)
    new_calibration_images, new_calibration_labels = create_monte_carlo_set(calibration_images, new_calibration_labels, m)
    new_calibration_set = create_JAX_dataset(new_calibration_images, new_calibration_labels)

    if save_model:
        save_model_params(state, run_directory + '/run_' + run_id + '.pkl')

    state = train_model(new_train_set, test_set, num_epochs=num_epochs, use_wandb=use_wandb, use_early_stopping=use_early_stopping, save_model=save_model)

    return state, new_train_set, new_calibration_set

def save_model_params(state, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'params': state['params'], 'batch_stats': state['batch_stats']}, f)

def load_model_params(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data['batch_stats']

def predict(state, images):
    cnn = CNN
    params = state['params']
    batch_stats = state['batch_stats']

    logits = cnn.apply(
        {'params': params, 'batch_stats': batch_stats},
        images,
        mutable=False,
        train=False
    )
    return nn.softmax(logits)