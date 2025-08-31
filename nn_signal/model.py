import torch.nn as nn
import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_markets, n_periods, n_features, n_labels,
                 market_embedding_dim=32, period_embedding_dim=16,
                 lstm_hidden=128, lstm_layers=2, attention_heads=8):
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
            nn.Linear(64, n_labels)
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
    
    def forward(self, sequence, mask, market_id, period):
        batch_size, seq_len = sequence.size(0), sequence.size(1)
        
        # Input projection
        x = self.input_projection(sequence)
        x = self.input_norm(x)
        
        # Apply mask to input (zero out invalid positions)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)  # (batch, seq, hidden)
            x = x * mask_expanded
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Apply mask to LSTM output
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
            lstm_out = lstm_out * mask_expanded
        
        # Self-attention with mask
        if mask is not None:
            # Create key padding mask (True = ignore, False = attend)
            key_padding_mask = (mask == 0)  # Invert mask for attention
            attn_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=key_padding_mask
            )
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Apply mask again after attention
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
            lstm_out = lstm_out * mask_expanded
        
        # Masked pooling strategies
        if mask is not None:
            # Get valid sequence lengths for each sample
            valid_lengths = mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
            valid_lengths = torch.clamp(valid_lengths, min=1.0)  # Avoid division by zero
            
            # Last valid output (not just last position)
            batch_indices = torch.arange(batch_size, device=sequence.device, dtype=torch.long)
            last_valid_indices = (mask.sum(dim=1) - 1).clamp(min=0).long()  # Convert to long
            last_output = lstm_out[batch_indices, last_valid_indices]  # (batch, hidden*2)
            
            # Masked max pooling
            masked_lstm = lstm_out.clone()
            masked_lstm[mask == 0] = float('-inf')  # Set invalid positions to -inf
            max_output = torch.max(masked_lstm, dim=1)[0]  # (batch, hidden*2)
            
            # Masked mean pooling
            masked_sum = (lstm_out * mask_expanded).sum(dim=1)  # (batch, hidden*2)
            mean_output = masked_sum / valid_lengths  # (batch, hidden*2)
            
        else:
            # Standard pooling without mask
            last_output = lstm_out[:, -1, :]
            max_output = torch.max(lstm_out, dim=1)[0]
            mean_output = torch.mean(lstm_out, dim=1)
        
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