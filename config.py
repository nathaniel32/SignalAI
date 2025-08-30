import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINED_PATH = "data/data.pth"
META_PATH = "data/meta.bin"
RETRAIN_MODEL = False
SAVE_MODEL = True

EPOCHS = 100
TRAIN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 50
HORIZON_LABEL = 15
HOLD_LABEL = False
THRESHOLD_LABEL = 0.0001
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 1e-5
SEQUENCE_CANDLE_LENGTH = 100
PRICE_COLUMNS = ['open', 'high', 'low', 'close']