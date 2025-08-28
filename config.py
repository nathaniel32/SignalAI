import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINED_PATH = "data/data.pth"
META_PATH = "data/meta.bin"
RETRAIN_MODEL = True
SAVE_MODEL = True

EPOCHS = 100
TRAIN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 50
SEQUENCE_LENGTH = 20
HORIZON_LABEL = 10
THRESHOLD_LABEL = 0.0001
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 500
NUM_LAYERS = 2
DROPOUT = 0.1