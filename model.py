import torch.nn as nn

# LSTM
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep output
        last_output = attn_out[:, -1, :]
        
        # Feature extraction
        features = self.feature_extractor(last_output)
        
        # Classification
        output = self.classifier(features)
        
        return output

""" 
import torch

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super(Model, self).__init__()
        
        # Multi-layer CNN with increasing channels
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.conv2 = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        
        self.conv3 = nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size*4)
        
        self.conv4 = nn.Conv1d(hidden_size*4, hidden_size*8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_size*8)
        
        # Additional deeper convolutions based on num_layers
        self.extra_convs = nn.ModuleList()
        for i in range(max(0, num_layers - 4)):
            self.extra_convs.append(nn.Sequential(
                nn.Conv1d(hidden_size*8, hidden_size*8, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size*8),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
        self.spatial_dropout = nn.Dropout2d(dropout * 0.5)
        
        # Multi-scale pooling for better feature extraction
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_size*8, hidden_size*4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size*4, hidden_size*8, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature dimension after pooling (avg + max pooling)
        feature_dim = hidden_size * 8 * 2
        
        # Multi-layer classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size*4),
            nn.BatchNorm1d(hidden_size*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size*4, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
            
            nn.Linear(hidden_size, num_classes)
        )
        
        # Residual connection for classifier
        self.residual_proj = nn.Linear(feature_dim, num_classes)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size) -> transpose to (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # First conv block
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Second conv block
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Third conv block
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Fourth conv block
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        # Extra conv layers if specified
        for extra_conv in self.extra_convs:
            residual = x
            x = extra_conv(x)
            x = x + residual  # Residual connection
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Multi-scale pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # (batch, hidden*8)
        max_pool = self.global_max_pool(x).squeeze(-1)  # (batch, hidden*8)
        
        # Concatenate pooled features
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden*16)
        
        # Classification with residual connection
        main_out = self.classifier(pooled_features)
        residual_out = self.residual_proj(pooled_features)
        
        # Combine main and residual outputs
        out = main_out + residual_out * 0.1  # Weighted residual
        
        return out """