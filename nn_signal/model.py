import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, n_markets, n_periods, sequence_length=20, n_features=4, 
                 market_embedding_dim=16, period_embedding_dim=8, 
                 lstm_hidden=64, lstm_layers=2):
        super(Model, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_hidden = lstm_hidden
        
        # Dual embedding
        self.market_embedding = nn.Embedding(n_markets, market_embedding_dim)
        self.period_embedding = nn.Embedding(n_periods, period_embedding_dim)
        
        # LSTM untuk sequence processing
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        # Dense layers
        combined_embedding_dim = market_embedding_dim + period_embedding_dim
        self.fc1 = nn.Linear(lstm_hidden + combined_embedding_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(32, 3)  # BUY, SELL, HOLD
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, sequence, market_id, period):
        batch_size = sequence.size(0)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(sequence)
        # Take last output
        lstm_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden)
        
        # Dual embedding
        market_emb = self.market_embedding(market_id)    # (batch_size, market_embedding_dim)
        period_emb = self.period_embedding(period)       # (batch_size, period_embedding_dim)
        
        # Concatenate semua features
        combined = torch.cat([lstm_out, market_emb, period_emb], dim=1)
        
        # Dense layers dengan residual-like connections
        x = self.leaky_relu(self.fc1(combined))
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return x