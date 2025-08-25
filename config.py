import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "data/data.json"
TRAINED_PATH = "data/data.pth"
META_PATH = "data/meta.bin"
RETRAIN_MODEL = False
SAVE_MODEL = True
EPOCHS = 200
TRAIN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 50
SEQUENCE_LENGTH = 20