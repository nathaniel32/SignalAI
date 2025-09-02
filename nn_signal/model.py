import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=1024, dropout=0):
        super(Model, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Enhanced input processing
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Positional encoding for sequence-like financial data
        self.pos_encoding = self._create_positional_encoding(input_size, hidden_size)
        
        # Multi-head self-attention for capturing complex dependencies
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer-like encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, dropout) for _ in range(3)
        ])
        
        # Feature extraction with residual connections
        self.feature_layers = nn.ModuleList([
            ResidualBlock(hidden_size, hidden_size, dropout),
            ResidualBlock(hidden_size, hidden_size // 2, dropout),
            ResidualBlock(hidden_size // 2, hidden_size // 4, dropout)
        ])
        
        # Market regime detection
        self.regime_detector = nn.Sequential(
            nn.Linear(hidden_size // 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 4),  # Bull, Bear, Sideways, Volatile
            nn.Softmax(dim=1)
        )
        
        # Risk-aware classification head
        self.risk_projector = nn.Sequential(
            nn.Linear(hidden_size // 4 + 4, hidden_size // 8),  # +4 for regime
            nn.LayerNorm(hidden_size // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3)
        )
        
        # Final classifier with uncertainty
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 8, hidden_size // 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_size // 16, num_classes)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size // 8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Feature importance weights
        self.feature_weights = nn.Parameter(torch.ones(input_size))
        
        # Apply custom initialization
        self.apply(self._init_weights)
        
    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He initialization for ReLU activations
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.MultiheadAttention):
            # Xavier initialization for attention
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply learned feature importance
        x_weighted = x * self.feature_weights.unsqueeze(0)
        
        # Project to hidden dimension
        x_proj = self.input_projection(x_weighted)
        x_proj = self.input_norm(x_proj)
        
        # Add positional encoding
        if x_proj.size(1) == self.pos_encoding.size(1):
            x_proj = x_proj + self.pos_encoding[:, :x_proj.size(1)]
        
        # Reshape for attention (batch, seq=1, features)
        x_seq = x_proj.unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.self_attention(x_seq, x_seq, x_seq)
        x_attended = attn_out.squeeze(1) + x_proj  # Residual connection
        
        # Apply transformer blocks
        x_encoded = x_attended
        for encoder in self.encoder_blocks:
            x_encoded = encoder(x_encoded)
        
        # Feature extraction with residuals
        features = x_encoded
        for layer in self.feature_layers:
            features = layer(features)
        
        # Market regime detection
        regime_probs = self.regime_detector(features)
        
        # Combine features with regime information
        combined_features = torch.cat([features, regime_probs], dim=1)
        risk_features = self.risk_projector(combined_features)
        
        # Classification
        logits = self.classifier(risk_features)
        confidence = self.confidence_estimator(risk_features)
        
        # Apply confidence weighting
        calibrated_logits = logits * (0.5 + confidence)  # Scale between 0.5-1.5
        
        return calibrated_logits


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture
        x_norm = self.norm1(x)
        x = x + self.feed_forward(x_norm)  # Residual connection
        x = self.norm2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Projection for residual connection if dimensions don't match
        self.projection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        residual = self.projection(x)
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        out = out + residual  # Residual connection
        out = F.relu(out, inplace=True)
        
        return out