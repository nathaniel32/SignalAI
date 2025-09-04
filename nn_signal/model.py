import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask shape: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v), attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class StockVisionTransformer(nn.Module):
    def __init__(self, n_markets, n_periods, n_features, n_labels,
                 d_model=256, n_heads=8, n_layers=6, d_ff=1024,
                 market_embedding_dim=64, period_embedding_dim=32,
                 dropout=0.1, max_length=1000):
        super().__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        
        # Input projection - convert features to d_model dimension
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Market and period embeddings
        self.market_embedding = nn.Embedding(n_markets, market_embedding_dim)
        self.period_embedding = nn.Embedding(n_periods, period_embedding_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Global average pooling untuk menggabungkan sequence
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        classifier_input_dim = d_model + market_embedding_dim + period_embedding_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_ff),
            nn.LayerNorm(d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_ff, d_ff // 2),
            nn.LayerNorm(d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            
            nn.Linear(d_ff // 2, d_ff // 4),
            nn.GELU(),
            nn.Dropout(dropout // 4),
            
            nn.Linear(d_ff // 4, n_labels)
        )
        
        # Temperature scaling untuk calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, sequence, mask, market_id, period):
        batch_size, seq_len, _ = sequence.shape
        
        # Project input to d_model dimension
        x = self.input_projection(sequence)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply mask to input
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x * mask_expanded
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        x = self.norm(x)
        
        # Global pooling dengan masking
        if mask is not None:
            # Masked global average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            
            # Sum valid tokens and divide by valid length
            valid_lengths = mask.sum(dim=1, keepdim=True).float().unsqueeze(-1)
            valid_lengths = torch.clamp(valid_lengths, min=1.0)
            
            pooled_output = x_masked.sum(dim=1) / valid_lengths.squeeze(-1)
        else:
            # Standard global average pooling
            pooled_output = x.mean(dim=1)  # (batch, d_model)
        
        # Get embeddings
        market_emb = self.market_embedding(market_id)  # (batch, market_embedding_dim)
        period_emb = self.period_embedding(period)    # (batch, period_embedding_dim)
        
        # Combine all features
        combined_features = torch.cat([pooled_output, market_emb, period_emb], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        return logits
    
    def get_attention_weights(self, sequence, mask, market_id, period):
        """
        Method untuk mengambil attention weights untuk visualisasi
        """
        batch_size, seq_len, _ = sequence.shape
        
        x = self.input_projection(sequence)
        x = self.pos_encoding(x)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x * mask_expanded
        
        attention_weights = []
        
        for transformer_block in self.transformer_blocks:
            # Get attention weights from each layer
            attn_output = transformer_block.attention(x, x, x, mask)
            # Store attention weights if needed for visualization
            x = transformer_block.norm1(x + transformer_block.dropout(attn_output))
            ff_output = transformer_block.feed_forward(x)
            x = transformer_block.norm2(x + ff_output)
        
        return attention_weights


def create_model(n_markets, n_periods, n_features, n_labels):
    if n_features <= 5:
        d_model = 256
    elif n_features <= 10:
        d_model = 384
    elif n_features <= 20:
        d_model = 512
    else:
        d_model = 768
    
    model = StockVisionTransformer(
        n_markets=n_markets,
        n_periods=n_periods, 
        n_features=n_features,
        n_labels=n_labels,
        d_model=d_model,
        n_heads=8,
        n_layers=6,
        d_ff=d_model * 4,
        dropout=0.1
    )
    
    return model