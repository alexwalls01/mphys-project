import pickle

ARCHITECTURE = 'LeNet'
NUM_CLASSES = 10
BATCH_SIZE = 32
IMAGE_SIZE = [32, 32]
RGB = True
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.0
NUM_EPOCHS = 100
PATIENCE = 5 # For early stopping
M = 10 # For Monte Carlo conformal prediction, only important to set here if using RT4U
WANDB_PROJECT = 'mccp-cnn-cifar-10h'
RUN_DIRECTORY = '/share/nas2_3/awalls/mccp/cifar-10h-runs'

# Set learning rate scheduler
LEARNING_RATE_SCHEDULER = False

# Parameters for warmup cosine decay scheduling
MIN_LEARNING_RATE = 1e-5
WARMUP_EPOCHS = 50

if RGB:
    INPUT_SHAPE = [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
else:
    INPUT_SHAPE = [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1]

config = {'architecture': ARCHITECTURE,
          'num_classes': NUM_CLASSES, 
          'num_epochs': NUM_EPOCHS, 
          'patience': PATIENCE,
          'batch_size': BATCH_SIZE,
          'image_size': IMAGE_SIZE, 
          'input_shape': INPUT_SHAPE, 
          'RGB': RGB, 
          'learning_rate': LEARNING_RATE,
          'min_learning_rate': MIN_LEARNING_RATE,
          'learning_rate_scheduler': LEARNING_RATE_SCHEDULER,
          'warmup_epochs': WARMUP_EPOCHS,
          'weight_decay': WEIGHT_DECAY,
          'dropout_rate': DROPOUT_RATE,
          'wandb_project': WANDB_PROJECT,
          'run_directory': RUN_DIRECTORY,
          'm': M,}

def update_CNN_config():
    with open('CNN_config.pkl', 'wb') as f:
        pickle.dump(config, f)

def load_CNN_config():
    with open('CNN_config.pkl', 'rb') as f:
        CNN_config = pickle.load(f)
    return CNN_config