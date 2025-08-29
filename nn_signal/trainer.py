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

    def main(self, datasets_df):
        try:
            train_df, val_df = datasets_df

            prepared_train_data, prepared_val_data, encoders = utils.prepare_data(train_df=train_df, val_df=val_df)

            X_sequences_train, X_market_ids_encoded_train, X_periods_encoded_train, Y_labels_encoded_train = prepared_train_data
            X_sequences_val, X_market_ids_encoded_val, X_periods_encoded_val, Y_labels_encoded_val = prepared_val_data
            encoder_market_ids, encoder_periods, encoder_labels = encoders

            n_train_data = len(X_sequences_train)
            n_val_data = len(X_sequences_val)
            total_data = n_train_data + n_val_data

            n_markets = len(encoder_market_ids.classes_)
            n_periods = len(encoder_periods.classes_)
            n_features = X_sequences_train.shape[2]
            n_labels = len(encoder_labels.classes_)

            model = nn_model.Model(n_markets, n_periods, n_features, n_labels).to(config.DEVICE)

            print("\nEncoder:")
            print("- Markets:", encoder_market_ids.classes_)
            print("- Periods:", encoder_periods.classes_)
            print("- Labels:", encoder_labels.classes_)

            print(f"\n- Device: {str(config.DEVICE)}")
            print(f'- Training Data: {n_train_data}')
            print(f'- Validation Data: {n_val_data}')
            print(f'- Total Data: {total_data}')

            summary(model, input_data=[
                torch.randn(1, config.SEQUENCE_CANDLE_LENGTH, n_features).to(config.DEVICE),
                torch.randint(0, n_markets, (1,)).to(config.DEVICE),
                torch.randint(0, n_periods, (1,)).to(config.DEVICE)
            ])

            total_params = sum(p.numel() for p in model.parameters())
            print("Parameters:", total_params)

            if total_data > 1:
                meta_data = {
                    "n_features": n_features,
                    "n_labels": n_labels,
                    'encoder_market_ids': encoder_market_ids,
                    "encoder_periods": encoder_periods,
                    "encoder_labels": encoder_labels
                }
                joblib.dump(meta_data, config.META_PATH)

                train_dataset = utils.DatasetManager(
                    X_sequences_train, X_market_ids_encoded_train,
                    X_periods_encoded_train, Y_labels_encoded_train
                )

                val_dataset = utils.DatasetManager(
                    X_sequences_val, X_market_ids_encoded_val,
                    X_periods_encoded_val, Y_labels_encoded_val
                )

                train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=config.TRAIN_BATCH_SIZE,
                                            shuffle=True,
                                            pin_memory=True)
                
                val_loader = DataLoader(dataset=val_dataset,
                                        batch_size=config.VALIDATION_BATCH_SIZE,
                                        shuffle=False,
                                        pin_memory=True)
                            
                if config.RETRAIN_MODEL:
                    try:
                        model.load_state_dict(torch.load(config.TRAINED_PATH, map_location=torch.device(config.DEVICE)))
                        print("\n!!!Retraining!!!")
                    except Exception as e:
                        print(f"\nError loading model for retraining: {e}")

                class_counts = np.bincount(Y_labels_encoded_train, minlength=n_labels)
                class_weights = len(Y_labels_encoded_train) / (len(class_counts) * class_counts)
                class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)

                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, mode="min")

                best_loss = float('inf')
                best_preds_array = []
                best_solution_array = []

                start_time = time.time()

                for epoch in range(config.EPOCHS):
                    train_loss = utils.train_fn(train_loader, model, optimizer, criterion)
                    val_loss, preds_array, solution_array = utils.val_fn(val_loader, model, criterion)
                    
                    scheduler.step(val_loss)
                    
                    print(f'\n== Epoch {epoch + 1}/{config.EPOCHS}')
                    print(f'Train Loss: {train_loss}')
                    print(f'Validation Loss: {val_loss}')
                    
                    if val_loss < best_loss and config.SAVE_MODEL:
                        best_preds_array = preds_array
                        best_solution_array = solution_array
                        best_loss = val_loss

                        os.makedirs(os.path.dirname(config.TRAINED_PATH), exist_ok=True)
                        torch.save(model.state_dict(), config.TRAINED_PATH)
                        print('new Model')

                    utils.show_conf_matrix(preds_array=preds_array, solution_array=solution_array, label=encoder_labels.classes_, title=f"{epoch + 1} Epochs | Total: {len(best_preds_array)}", plot=best_loss==val_loss)
                end_time = time.time()

                trainingsdauer = (end_time - start_time) / 60

                print(f"\nTraining duration: {trainingsdauer:.2f} minutes")

                #utils.show_conf_matrix(preds_array=best_preds_array, solution_array=best_solution_array, label=encoder_labels.classes_, title=f"{best_epochs} Epochs | Total: {len(best_preds_array)} | Best Model", plot=True)
            else:
                print("\n!!!Too little X_sequences to train AI!!!")
        except Exception as e:
            raise FileNotFoundError(f"\nError in the training process")