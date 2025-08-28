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
import numpy as np

class Trainer:
    def __init__(self):
        pass

    def main(self, df):
        try:
            prepared_data, encoders = utils.prepare_data(df)

            X_sequences, X_market_ids_encoded, X_periods_encoded, Y_labels_encoded = prepared_data
            encoder_market_ids, encoder_periods, encoder_labels = encoders
       
            print("Markets:", encoder_market_ids.classes_)
            print("Periods:", encoder_periods.classes_)
            print("Labels:", encoder_labels.classes_)

            meta_data = {
                "n_features": X_sequences.shape[2],
                'encoder_market_ids': encoder_market_ids,
                "encoder_periods": encoder_periods,
                "encoder_labels": encoder_labels
            }
            joblib.dump(meta_data, config.META_PATH)

            if len(X_sequences) > 1:
                # Split data
                indices = np.arange(len(X_sequences))
                train_idx, val_idx = train_test_split(
                    indices, test_size=0.2, random_state=42, stratify=Y_labels_encoded
                )
                
                train_dataset = utils.DatasetManager(
                    X_sequences[train_idx], X_market_ids_encoded[train_idx],
                    X_periods_encoded[train_idx], Y_labels_encoded[train_idx]
                )

                val_dataset = utils.DatasetManager(
                    X_sequences[val_idx], X_market_ids_encoded[val_idx],
                    X_periods_encoded[val_idx], Y_labels_encoded[val_idx]
                )

                train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=config.TRAIN_BATCH_SIZE,
                                            shuffle=True,
                                            pin_memory=True)
                
                val_loader = DataLoader(dataset=val_dataset,
                                        batch_size=config.VALIDATION_BATCH_SIZE,
                                        shuffle=False,
                                        pin_memory=True)
                            
                n_markets = len(encoder_market_ids.classes_)
                n_periods = len(encoder_periods.classes_)
                model = nn_model.Model(n_markets, n_periods, X_sequences.shape[2]).to(config.DEVICE)

                total_params = sum(p.numel() for p in model.parameters())
                print("Parameters:", total_params)

                if config.RETRAIN_MODEL:
                    try:
                        model.load_state_dict(torch.load(config.TRAINED_PATH, map_location=torch.device(config.DEVICE)))
                        print("\n!!!Retraining!!!")
                    except Exception as e:
                        print(f"\nError loading model for retraining: {e}")

                class_counts = np.bincount(Y_labels_encoded)
                class_weights = len(Y_labels_encoded) / (len(class_counts) * class_counts)
                class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)

                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

                best_loss = float('inf')
                best_epochs = 1
                best_preds_array = []
                best_solution_array = []

                print("=" * 20)
                print("Device: " + str(config.DEVICE))
                print(f'Training Data: {len(train_idx)}')
                print(f'Validation Data: {len(val_idx)}')
                print(f'Total Data: {len(indices)}')

                start_time = time.time()

                for epoch in range(config.EPOCHS):
                    train_loss = utils.train_fn(train_loader, model, optimizer, criterion)
                    val_loss, preds_array, solution_array = utils.val_fn(val_loader, model, criterion)
                    
                    scheduler.step(val_loss)
                    
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
                summary(model, input_data=[
                    torch.randn(1, config.SEQUENCE_CANDLE_LENGTH, 4).to(config.DEVICE),
                    torch.randint(0, n_markets, (1,)).to(config.DEVICE),
                    torch.randint(0, n_periods, (1,)).to(config.DEVICE)
                ])
            else:
                print("\n!!!Too little X_sequences to train AI!!!")
        except Exception as e:
            raise FileNotFoundError(f"\nError in the training process")