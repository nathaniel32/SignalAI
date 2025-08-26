""" 
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

""" import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super(Model, self).__init__()
        
        # Simplified CNN with proper channel progression
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.conv2 = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        
        self.conv3 = nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size*2)
        
        # Pooling layers for dimensionality reduction
        self.pool = nn.MaxPool1d(2, stride=2)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simplified classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size//2, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Second conv block  
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Global pooling
        x = self.global_avg_pool(x).squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        
        return x """

""" import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feed forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                 padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, channels, _ = x.size()
        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y).view(batch, channels, 1)
        return x * y

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_size) * 0.02)
        
        # Multi-scale CNN feature extraction
        self.conv_blocks = nn.ModuleList()
        channels = [hidden_size, hidden_size*2, hidden_size*4, hidden_size*6]
        
        for i in range(len(channels)-1):
            block = nn.Sequential(
                DepthwiseSeparableConv1D(channels[i], channels[i+1], 
                                       kernel_size=3, padding=1),
                nn.BatchNorm1d(channels[i+1]),
                nn.GELU(),
                SEBlock(channels[i+1]),
                nn.Dropout(dropout * (i+1) * 0.1)
            )
            self.conv_blocks.append(block)
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(sum(channels), hidden_size*4, 1),
            nn.BatchNorm1d(hidden_size*4),
            nn.GELU()
        )
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size*4, num_heads=8, d_ff=hidden_size*8, dropout=dropout)
            for _ in range(min(num_layers, 6))
        ])
        
        # Multi-head pooling
        self.pooling_heads = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),
            nn.AdaptiveAvgPool1d(2),
        ])
        
        # Dynamic feature selection
        total_features = hidden_size*4 * 4  # 3 pooling methods + avg of adaptive(2)
        self.feature_selector = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Linear(total_features // 2, total_features),
            nn.Sigmoid()
        )
        
        # Final classifier with multiple branches
        self.main_classifier = nn.Sequential(
            nn.Linear(total_features, hidden_size*4),
            nn.BatchNorm1d(hidden_size*4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size*4, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            
            nn.Linear(hidden_size, num_classes)
        )
        
        # Auxiliary classifier for regularization
        self.aux_classifier = nn.Sequential(
            nn.Linear(total_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_aux=False):
        batch_size, seq_len, _ = x.size()
        
        # Input projection and positional encoding
        x = self.input_proj(x)
        if seq_len <= self.pos_encoding.size(0):
            x += self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transpose for conv layers: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Multi-scale CNN feature extraction
        conv_features = [x]
        current_x = x
        
        for conv_block in self.conv_blocks:
            current_x = conv_block(current_x)
            conv_features.append(current_x)
        
        # Feature fusion
        fused_features = torch.cat(conv_features, dim=1)
        x = self.feature_fusion(fused_features)
        
        # Back to (batch, seq_len, features) for transformer
        x = x.transpose(1, 2)
        
        # Transformer processing
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Back to (batch, features, seq_len) for pooling
        x = x.transpose(1, 2)
        
        # Multi-head pooling
        pooled_features = []
        
        # Average and max pooling
        pooled_features.append(self.pooling_heads[0](x).squeeze(-1))
        pooled_features.append(self.pooling_heads[1](x).squeeze(-1))
        
        # Adaptive pooling to 2 and take mean
        adaptive_pool = self.pooling_heads[2](x)  # (batch, features, 2)
        pooled_features.append(adaptive_pool.mean(dim=-1))  # Average over the 2 positions
        
        # Statistical pooling
        std_pool = torch.std(x, dim=-1)
        pooled_features.append(std_pool)
        
        # Concatenate all pooled features
        final_features = torch.cat(pooled_features, dim=1)
        
        # Dynamic feature selection
        feature_weights = self.feature_selector(final_features)
        final_features = final_features * feature_weights
        
        # Main classification
        main_output = self.main_classifier(final_features)
        
        if return_aux:
            aux_output = self.aux_classifier(final_features)
            return main_output, aux_output
        
        return main_output """

import torch.nn as nn
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layers with gradient clipping friendly initialization
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
        
        # Global pooling instead of just last timestep
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),  # BatchNorm instead of LayerNorm
            nn.GELU(),  # GELU instead of ReLU
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Classification head with better regularization
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
        
        # Global average pooling instead of just last timestep
        # Shape: (batch_size, seq_len, hidden_size*2) -> (batch_size, hidden_size*2, seq_len)
        pooled = self.global_pool(attn_out.transpose(1, 2))
        # Shape: (batch_size, hidden_size*2, 1) -> (batch_size, hidden_size*2)
        pooled = pooled.squeeze(-1)
        
        # Feature extraction
        features = self.feature_extractor(pooled)
        
        # Classification
        logits = self.classifier(features)
        
        return logits