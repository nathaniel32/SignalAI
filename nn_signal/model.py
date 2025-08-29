import torch.nn as nn
import torch
import torch.nn.functional as F

""" class Model(nn.Module):
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
        
        return x """

class Model(nn.Module):
   def __init__(self, n_markets, n_periods, n_features, n_labels,
                market_embedding_dim=32, period_embedding_dim=16,
                lstm_hidden=4, lstm_layers=1, attention_heads=8):
       super(Model, self).__init__()
       
       self.n_features = n_features
       self.lstm_hidden = lstm_hidden
       
       # Enhanced embeddings
       self.market_embedding = nn.Embedding(n_markets, market_embedding_dim)
       self.period_embedding = nn.Embedding(n_periods, period_embedding_dim)
       
       # Input projection
       self.input_projection = nn.Linear(n_features, lstm_hidden)
       self.input_norm = nn.LayerNorm(lstm_hidden)
       
       # Bidirectional LSTM
       self.lstm = nn.LSTM(
           input_size=lstm_hidden,
           hidden_size=lstm_hidden,
           num_layers=lstm_layers,
           batch_first=True,
           dropout=0.3 if lstm_layers > 1 else 0,
           bidirectional=True
       )
       
       # Multi-head attention
       self.attention = nn.MultiheadAttention(
           embed_dim=lstm_hidden * 2,
           num_heads=attention_heads,
           dropout=0.5,
           batch_first=True
       )
       self.attention_norm = nn.LayerNorm(lstm_hidden * 2)
       
       # Calculate correct combined dimension
       # last_output + max_output + mean_output = 3 * (lstm_hidden * 2)
       sequence_dim = lstm_hidden * 2 * 3  # 768 for lstm_hidden=128
       combined_dim = sequence_dim + market_embedding_dim + period_embedding_dim
       
       # Feature fusion
       self.feature_fusion = nn.Sequential(
           nn.Linear(combined_dim, 256),
           nn.LayerNorm(256),
           nn.GELU(),
           nn.Dropout(0.3)
       )
       
       # Residual blocks
       self.residual_blocks = nn.ModuleList([
           self._make_residual_block(256, 256) for _ in range(2)
       ])
       
       # Output classifier
       self.classifier = nn.Sequential(
           nn.Linear(256, 128),
           nn.LayerNorm(128),
           nn.GELU(),
           nn.Dropout(0.2),
           nn.Linear(128, 64),
           nn.GELU(),
           nn.Dropout(0.1),
           nn.Linear(64, n_labels)  # BUY, SELL, HOLD
       )
       
       # Temperature for calibration
       self.temperature = nn.Parameter(torch.ones(1))
       
       self._init_weights()
   
   def _make_residual_block(self, in_dim, out_dim):
       return nn.Sequential(
           nn.Linear(in_dim, out_dim),
           nn.LayerNorm(out_dim),
           nn.GELU(),
           nn.Dropout(0.2),
           nn.Linear(out_dim, out_dim),
           nn.LayerNorm(out_dim)
       )
   
   def _init_weights(self):
       for module in self.modules():
           if isinstance(module, nn.Linear):
               nn.init.xavier_uniform_(module.weight)
               if module.bias is not None:
                   nn.init.constant_(module.bias, 0)
           elif isinstance(module, nn.LSTM):
               for name, param in module.named_parameters():
                   if 'weight' in name:
                       nn.init.orthogonal_(param)
                   elif 'bias' in name:
                       nn.init.constant_(param, 0)
           elif isinstance(module, nn.Embedding):
               nn.init.normal_(module.weight, 0, 0.1)
   
   def forward(self, sequence, market_id, period):
       batch_size = sequence.size(0)
       
       # Input projection
       x = self.input_projection(sequence)
       x = self.input_norm(x)
       
       # Bidirectional LSTM
       lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
       
       # Self-attention
       attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
       lstm_out = self.attention_norm(lstm_out + attn_out)
       
       # Multiple pooling strategies
       last_output = lstm_out[:, -1, :]  # (batch, hidden*2)
       max_output = torch.max(lstm_out, dim=1)[0]  # (batch, hidden*2)
       mean_output = torch.mean(lstm_out, dim=1)   # (batch, hidden*2)
       
       # Concatenate pooled outputs
       sequence_features = torch.cat([last_output, max_output, mean_output], dim=1)
       
       # Embeddings
       market_emb = self.market_embedding(market_id)
       period_emb = self.period_embedding(period)
       
       # Combine all features
       combined = torch.cat([sequence_features, market_emb, period_emb], dim=1)
       
       # Feature fusion
       x = self.feature_fusion(combined)
       
       # Residual blocks
       for block in self.residual_blocks:
           residual = x
           x = block(x) + residual
           x = F.gelu(x)
       
       # Final classification
       logits = self.classifier(x)
       logits = logits / self.temperature
       
       return logits