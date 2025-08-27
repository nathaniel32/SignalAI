import os
import time
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib
import nn_signal.utils as utils
import nn_signal.model as nn_model
import config
from torchinfo import summary
import logging

logging.basicConfig(level=logging.INFO)

class Trainer:
    def __init__(self):
        pass

    def main(self, df):
        try:
            features, labels, encoder_labels = utils.prepare_data(df)
            
            logging.debug(features.shape)
            logging.debug(features)
            logging.debug(labels)
            logging.debug(encoder_labels.classes_)

            if len(features) > 1:
                train_data, val_data, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels) #stratify=labels
            
                train_dataset = utils.DatasetManager(features=train_data, labels=train_labels, sequence_length=config.SEQUENCE_LENGTH)
                train_data_loader = DataLoader( dataset=train_dataset,
                                                batch_size=config.TRAIN_BATCH_SIZE,
                                                shuffle=True,
                                                pin_memory=True)
                
                val_dataset = utils.DatasetManager(features=val_data, labels=val_labels, sequence_length=config.SEQUENCE_LENGTH)
                val_data_loader = DataLoader(   dataset=val_dataset,
                                                batch_size=config.VALIDATION_BATCH_SIZE,
                                                shuffle=False,
                                                pin_memory=True)
                
                logging.info("\n\nTrain data: %d\nVal data: %d\nTrain labels: %d\nVal labels: %d", len(train_data), len(val_data), len(train_labels), len(val_labels))
            
                device = config.DEVICE 
                model = nn_model.Model(
                    input_size=features.shape[1],
                    hidden_size=config.HIDDEN_SIZE,
                    num_layers=config.NUM_LAYERS,
                    dropout=config.DROPOUT,
                    num_classes=len(encoder_labels.classes_)
                )
                model.to(device)

                meta_data = {
                    'encoder_labels': encoder_labels,
                    "input_size": features.shape[1],
                    "hidden_size": config.HIDDEN_SIZE,
                    "num_layers": config.NUM_LAYERS,
                    "dropout": config.DROPOUT,
                    "num_classes": len(encoder_labels.classes_)
                }
                joblib.dump(meta_data, config.META_PATH)

                if config.RETRAIN_MODEL:
                    try:
                        model.load_state_dict(torch.load(config.TRAINED_PATH, map_location=torch.device(config.DEVICE)))
                        print("\n!!!Retraining!!!")
                    except Exception as e:
                        print(f"\nError loading model for retraining: {e}")

                optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

                best_loss = float('inf')
                best_epochs = 1
                best_preds_array = []
                best_solution_array = []

                print("=" * 20)
                print("Device: " + str(device))
                print(f'Training Data: {len(train_data)}')
                print(f'Validation Data: {len(val_data)}')
                print(f'Total Data: {len(features)}')

                start_time = time.time()

                for epoch in range(config.EPOCHS):
                    train_loss = utils.train_fn(train_data_loader, model, optimizer, device)
                    val_loss, preds_array, solution_array = utils.val_fn(val_data_loader, model, device)

                    print(f'\n== Epoch {epoch + 1}/{config.EPOCHS}')
                    print(f'Train Loss: {train_loss}')
                    print(f'Validation Loss: {val_loss}')
                    
                    if val_loss < best_loss or epoch % 100 == 0 or epoch == config.EPOCHS-1:
                        best_preds_array = preds_array
                        best_solution_array = solution_array
                        utils.show_conf_matrix(preds_array=preds_array, solution_array=solution_array, label=encoder_labels.classes_, title="AI")
                        
                        if val_loss < best_loss and config.SAVE_MODEL:
                            os.makedirs(os.path.dirname(config.TRAINED_PATH), exist_ok=True)
                            torch.save(model.state_dict(), config.TRAINED_PATH)
                            best_loss = val_loss
                            best_epochs = epoch + 1
                            print('new Model')

                end_time = time.time()

                trainingsdauer = (end_time - start_time) / 60

                print(f"\nTraining duration: {trainingsdauer:.2f} minutes")

                utils.show_conf_matrix(preds_array=best_preds_array, solution_array=best_solution_array, label=encoder_labels.classes_, title=f"{best_epochs} Epochs | Total: {len(best_preds_array)} | AI", plot=True)
                summary(model, input_size=(config.TRAIN_BATCH_SIZE, config.SEQUENCE_LENGTH, features.shape[1]))
            else:
                print("\n!!!Too little features to train AI!!!")
        except Exception as e:
            raise FileNotFoundError(f"\nError in the training process")