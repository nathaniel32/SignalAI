import os
import time
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib
import utils
import model as nn_model
import config
from torchinfo import summary
import logging

logging.basicConfig(level=logging.INFO)

def train():
    try:
        features, labels, encoder_labels = utils.get_data(config.DATASET_PATH)
        
        logging.info(features.shape)
        #logging.info(features)
        #logging.info(labels)
        #logging.info(encoder_labels.classes_)

        if len(features) > 1:
            meta_data = {
                'encoder_labels': encoder_labels
            }
            joblib.dump(meta_data, config.META_PATH)

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
            
            logging.info("\nTrain data: %d\nVal data: %d\nTrain labels: %d\nVal labels: %d", len(train_data), len(val_data), len(train_labels), len(val_labels))
        
            device = config.DEVICE 
            model = nn_model.Model(
                input_size=features.shape[1],
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                num_classes=len(encoder_labels.classes_)
            )
            model.to(device)

            if config.RETRAIN_MODEL:
                try:
                    model.load_state_dict(torch.load(config.TRAINED_PATH, map_location=torch.device(config.DEVICE)))
                    print("\n!!!Retraining!!!")
                except Exception as e:
                    print(f"\nError loading model for retraining: {e}")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_loss = float('inf')
            best_epochs = 1
            best_preds_array = []
            best_solution_array = []

            print("=" * 10)
            print("Device: " + str(device))
            print(f'Training Data: {len(train_data)}')
            print(f'Validation Data: {len(val_data)}')
            print(f'Total Data: {len(features)}')

            start_time = time.time()

            for epoch in range(config.EPOCHS):
                train_loss = utils.train_fn(train_data_loader, model, optimizer, device)
                val_loss, preds_array, solution_array = utils.val_fn(val_data_loader, model, device)

                if val_loss < best_loss or epoch % 100 == 0 or epoch == config.EPOCHS-1:
                    best_preds_array = preds_array
                    best_solution_array = solution_array
                    print(encoder_labels.classes_)
                    utils.show_conf_matrix(preds_array=preds_array, solution_array=solution_array, label=encoder_labels.classes_, title="AI")
                    print(f'\n== Epoch {epoch + 1}/{config.EPOCHS}')
                    print(f'Train Loss: {train_loss}')
                    
                    if val_loss < best_loss and config.SAVE_MODEL:
                        os.makedirs(os.path.dirname(config.TRAINED_PATH), exist_ok=True)
                        torch.save(model.state_dict(), config.TRAINED_PATH)
                        best_loss = val_loss
                        best_epochs = epoch + 1
                        print(f'Validation Loss: {best_loss}, new Model')
                    else:
                        print(f'Validation Loss: {val_loss}')

            end_time = time.time()

            trainingsdauer = (end_time - start_time) / 60

            print(f"\nTraining duration: {trainingsdauer:.2f} minutes")

            utils.show_conf_matrix(preds_array=best_preds_array, solution_array=best_solution_array, label=encoder_labels.classes_, title=f"{best_epochs} Epochs | Total: {len(best_preds_array)} | AI", plot=True)
            summary(model, input_size=(config.TRAIN_BATCH_SIZE, 20, features.shape[1]))
        else:
            print("\n!!!Too little features to train AI!!!")
    except Exception as e:
        raise FileNotFoundError(f"\nError in the training process")
    
train()