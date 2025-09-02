import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, dropout=0.3):
        super(Model, self).__init__()
        
        self.num_classes = num_classes
        
        # Normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
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
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        batch_size, input_size = x.size()
        
        # Input normalization
        x = self.input_norm(x)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Classification
        logits = self.classifier(features)
        return logits