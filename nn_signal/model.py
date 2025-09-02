import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # No dropout for single layer
            batch_first=True,
            bidirectional=True
        )
        
        # Self-attention with residual connection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=4,  # Reduced from 8 to prevent overfitting
            dropout=dropout * 0.5,  # Lighter dropout in attention
            batch_first=True
        )
        
        # Attention output normalization
        self.attn_norm = nn.LayerNorm(hidden_size * 2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm
            nn.GELU(),  # GELU instead of ReLU
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization with gain
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)  # Small positive bias
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)  # Orthogonal for recurrent weights
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)  # Residual connection
        
        # Global average pooling
        # Shape: (batch_size, seq_len, hidden_size*2) -> (batch_size, hidden_size*2, seq_len)
        pooled = self.global_pool(attn_out.transpose(1, 2))
        # Shape: (batch_size, hidden_size*2, 1) -> (batch_size, hidden_size*2)
        pooled = pooled.squeeze(-1)
        
        # Feature extraction
        features = self.feature_extractor(pooled)
        
        # Classification
        logits = self.classifier(features)
        
        return logits