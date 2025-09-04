import joblib
import torch
import nn_signal.utils as utils
import nn_signal.model as nn_model
import config
import os
import numpy as np

class Predictor:
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"\nYou need to train AI on main.py")
        self.meta_data = joblib.load(config.META_PATH)
        
        self.n_features = self.meta_data['n_features']
        self.encoder_market_ids = self.meta_data['encoder_market_ids']
        self.encoder_periods = self.meta_data['encoder_periods']
        self.encoder_labels = self.meta_data['encoder_labels']
        self.model = nn_model.create_model(n_markets=len(self.encoder_market_ids.classes_), n_periods=len(self.encoder_periods.classes_), n_features=self.n_features, n_labels=len(self.encoder_labels.classes_)).to(config.DEVICE)
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(config.DEVICE)))
        self.model.to(config.DEVICE).eval()
    
    def prediction(self, sequence, mask, market_encoded, period_encoded):
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(config.DEVICE)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(config.DEVICE)
        market_tensor = torch.LongTensor([market_encoded]).to(config.DEVICE)
        period_tensor = torch.LongTensor([period_encoded]).to(config.DEVICE)
        with torch.no_grad():
            logits = self.model(sequence_tensor, mask_tensor, market_tensor, period_tensor)
        return logits
    
    def logits_extraction(self, logits):
        class_scores, class_preds = utils.to_yhat(logits)
        return [self.encoder_labels.classes_, class_scores[0]], class_preds
    
    def main(self, df, market_id, period):
        try:
            market_encoded = self.encoder_market_ids.transform([market_id])[0]
        except (AttributeError, ValueError):
            raise RuntimeError("Market encoder not trained yet or label unknown")

        try:
            period_encoded = self.encoder_periods.transform([period])[0]
        except (AttributeError, ValueError):
            raise RuntimeError("Period encoder not trained yet or label unknown")

        try:
            df_last_sequence = df.tail(config.SEQUENCE_CANDLE_LENGTH-config.HORIZON_LABEL)
            df_last_sequence, ai_indicators = utils.create_indicators(df=df_last_sequence)
            df_last_sequence, valid_mask = utils.create_mask(df=df_last_sequence, features=ai_indicators)

            sequence = df_last_sequence[ai_indicators].values
            mask = df_last_sequence['mask'].values
            
            sequence = np.nan_to_num(sequence, nan=0.0)
            #print(df)
            #print(sequence)
            #print(mask)

            # last row
            print("\nTimestamp: ", df.tail(1).index[0])
            
            logits = self.prediction(sequence, mask, market_encoded, period_encoded)
            return self.logits_extraction(logits)
        except Exception as e:
            print(e)