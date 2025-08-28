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
        self.meta_data = joblib.load(config.META_PATH)
        
        self.encoder_market_ids = self.meta_data['encoder_market_ids']
        self.encoder_periods = self.meta_data['encoder_periods']
        self.encoder_labels = self.meta_data['encoder_labels']
        
        self.model = nn_model.Model(
            len(self.encoder_market_ids.classes_),
            len(self.encoder_periods.classes_)
        )
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(config.DEVICE)))
        self.model.to(config.DEVICE).eval()
    
    def prediction(self, sequence, market_encoded, period_encoded):
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(config.DEVICE)
        market_tensor = torch.LongTensor([market_encoded]).to(config.DEVICE)
        period_tensor = torch.LongTensor([period_encoded]).to(config.DEVICE)
        with torch.no_grad():
            logits = self.model(sequence_tensor, market_tensor, period_tensor)
        return logits
    
    def logits_extraction(self, logits):
        class_scores, class_preds = utils.to_yhat(logits)
        return class_preds, [self.encoder_labels.classes_, class_scores[0]]
                
    def main(self, df, market_id, period):        
        sequence = df[config.PRICE_COLUMNS].tail(config.SEQUENCE_CANDLE_LENGTH).values
        market_encoded = self.encoder_market_ids.transform([market_id])[0]
        period_encoded = self.encoder_periods.transform([period])[0]

        # last row
        print("Timestamp: ", df.tail(1).index[0])
        
        logits = self.prediction(sequence, market_encoded, period_encoded)
        return self.logits_extraction(logits)