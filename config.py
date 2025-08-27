import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH = "data/data.json"
TRAINED_PATH = "data/data.pth"
META_PATH = "data/meta.bin"
RETRAIN_MODEL = False
SAVE_MODEL = True
EPOCHS = 200
TRAIN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 50
SEQUENCE_LENGTH = 20
HORIZON_LABEL = 10
THRESHOLD_LABEL = 0.001
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 1000
NUM_LAYERS = 2
DROPOUT = 0.5