import joblib
import torch
import nn_signal.utils as utils
import nn_signal.model as nn_model
import config
import os
import json
import pandas as pd

class Predictor:
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"\nYou need to train AI on main.py")
        self.device = config.DEVICE
        self.meta_data = joblib.load(config.META_PATH)
        
        self.encoder_labels = self.meta_data['encoder_labels']
        
        self.num_benutzerids = len(self.encoder_labels.classes_)

        self.model = nn_model.Model(
            input_size=self.meta_data['input_size'],
            hidden_size=self.meta_data['hidden_size'],
            num_layers=self.meta_data['num_layers'],
            dropout=self.meta_data['dropout'],
            num_classes=self.meta_data['num_classes']
        )
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(config.DEVICE)))
        self.model.to(self.device).eval()
    
    def prediction(self, features):
        with torch.no_grad():
            logits = self.model(features)
        return logits
    
    def logits_extraction(self, logits):
        class_scores, class_preds = utils.to_yhat(logits)
        return class_preds, [self.encoder_labels.classes_, class_scores[0]]
                
    def main(self, df):
        
        print(df)
        _, ai_features = utils.create_advanced_features(df=df)

        print(df)
        df = df[ai_features].apply(pd.to_numeric, errors="coerce").dropna(axis=0)
        
        # last row
        df = df.tail(1)
        print("Timestamp: ", df.index[0])

        feature = df.values[0]
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
        logits = self.prediction(feature_tensor)
        return self.logits_extraction(logits)